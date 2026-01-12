import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    PreTrainedModel
)
from transformers.trainer_pt_utils import nested_detach
from peft import get_peft_model, LoraConfig
import numpy as np
import json
import os
from datetime import datetime

import argparse

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    if args.config:
        import json
        with open(args.config, "r") as f:
            config = json.load(f)
        return config
    return None

# Try importing user utils
try:
    from yurain_utils import *
except ImportError:
    def read_json(path):
        with open(path, 'r') as f:
            return json.load(f)

# ============================================================================
# Set Random Seed
# ============================================================================

import torch
import numpy as np
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ================= Config Parameters =================
class GRPOConfig:
    def __init__(self):
        self.model_name = "./models/Qwen2.5_3B_Base"  # Placeholder
        self.max_length = 1536
        self.group_size = 6
        self.beta = 0
        self.epsilon = 0.2    
        self.epsilon_low = 0.2
        self.epsilon_high = 0.24
        self.learning_rate = 5e-7
        
        self.per_device_batch_size = 2  # Prompts per device
        self.gradient_accumulation_steps = 10  # Gradient accumulation steps
        self.log_prob_micro_batch_size_per_gpu = 2  # Batch size for calculating old log probs
        self.ppo_micro_batch_size_per_gpu = 6
        self.epochs = 1
        
        self.dataset_path = "./data/training_data.json" # Placeholder
        
        self.use_lora = False
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_target_modules = ["q_proj", "v_proj"]
        
        self.output_dir = "./grpo_model"
        self.log_file = os.path.join(self.output_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        # Acceleration Config
        self.use_flash_attn = True  # Enable Flash Attention 2
        self.use_gradient_checkpointing = True # Suggested False if memory allows, for speed

# ================= 1. Dataset Definition (Token-level Log Probs) =================
class GRPODataset(Dataset):
    """
    Dataset that pre-computes token-level old_log_probs
    """
    def __init__(self, data, tokenizer, max_length=2048, group_size=16, 
                 model=None, mini_batch_size=4):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.group_size = group_size
        
        # Preprocess all data
        self.processed_data = []
        for item in data:
            processed = self._process_item(item)
            self.processed_data.append(processed)
        
        # Precompute old_log_probs (token-level)
        if model is not None:
            self._precompute_old_log_probs(model, mini_batch_size)
    
    def _process_item(self, item):
        prompt = item["prompt"]
        completions = item["completions"]
        rewards = item["rewards"]

        assert len(completions) == len(rewards), "Completions and rewards length mismatch"
        curr_len = len(completions)
        indices = list(range(curr_len))

        if curr_len < self.group_size:
            # Pad: Circular copy, maintain original distribution as much as possible
            # Compared to random sampling, this maintains reward mean/variance stability better
            multiplier = (self.group_size // curr_len) + 1
            extended_indices = (indices * multiplier)[:self.group_size]
            indices = extended_indices
            
        elif curr_len > self.group_size:
            # Truncate: Prioritize keeping distinct reward values to ensure intra-group variance
            # 1. Group by reward value
            reward_map = {} 
            for idx, r in enumerate(rewards):
                if r not in reward_map:
                    reward_map[r] = []
                reward_map[r].append(idx)
            
            # 2. Stratified Sampling
            unique_rewards = sorted(reward_map.keys())
            
            # Shuffle within each bucket ensures random selection of same-score samples
            for r in unique_rewards:
                random.shuffle(reward_map[r])
            
            selected_indices = []
            # Round-robin pick one from each reward group until full
            while len(selected_indices) < self.group_size:
                added = False
                for r in unique_rewards:
                    if len(selected_indices) >= self.group_size:
                        break
                    if reward_map[r]:
                        selected_indices.append(reward_map[r].pop())
                        added = True
                if not added:
                    break # All samples taken
            
            indices = selected_indices
            
            # Shuffle again to avoid patterns from round-robin
            random.shuffle(indices)
        
        completions = [completions[i] for i in indices]
        rewards = [rewards[i] for i in indices]

        input_ids_list = []
        attention_mask_list = []
        loss_mask_list = []

        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)

        for completion in completions:
            full_text = prompt + completion
            tokenized = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            
            loss_mask = torch.zeros_like(input_ids)
            
            # === Mod Start ===
            # Find start of valid content (skip left padding)
            # First position where attention_mask is 1 is start of Prompt
            valid_indices = torch.nonzero(attention_mask)
            if valid_indices.numel() > 0:
                start_idx = valid_indices[0].item()
            else:
                start_idx = 0
            
            # Completion start = Prompt start + Prompt len
            completion_start_idx = start_idx + prompt_len
            
            if completion_start_idx < self.max_length:
                loss_mask[completion_start_idx:] = 1 
            # === Mod End ===

            loss_mask = loss_mask * attention_mask

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            loss_mask_list.append(loss_mask)

        return {
            "input_ids": torch.stack(input_ids_list),           # [G, Seq]
            "attention_mask": torch.stack(attention_mask_list), # [G, Seq]
            "loss_mask": torch.stack(loss_mask_list),           # [G, Seq]
            "rewards": torch.tensor(rewards, dtype=torch.float32), # [G]
            "old_token_log_probs": None,  # [G, Seq-1] token-level log probs
        }
    
    @torch.no_grad()
    def _precompute_old_log_probs(self, model, mini_batch_size):
        """Precompute token-level old_log_probs using initial model (supports multi-GPU)"""
        
        # 1. Env check and init
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            device = torch.device(f"cuda:{local_rank}")
        else:
            world_size = 1
            local_rank = 0
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Rank {local_rank}] Pre-computing token-level old log probs on device {device}...")
        
        # 2. Move model to current GPU
        model = model.to(device)
        model.eval()
        
        # 3. Assign tasks: Each Rank computes a part of data
        my_indices = [i for i in range(len(self.processed_data)) if i % world_size == local_rank]
        my_old_token_log_probs = []  # Buffer [N_local, G, Seq-1]
        
        for i, idx in enumerate(my_indices):
            item = self.processed_data[idx]
            input_ids = item["input_ids"].to(device)      # [G, Seq]
            attention_mask = item["attention_mask"].to(device)
            loss_mask = item["loss_mask"].to(device)
            
            g_size = input_ids.shape[0]
            token_log_probs_list = []
            
            # Batch computation to save memory
            for j in range(0, g_size, mini_batch_size):
                end_idx = min(j + mini_batch_size, g_size)
                mini_input_ids = input_ids[j:end_idx]
                mini_attention_mask = attention_mask[j:end_idx]
                mini_loss_mask = loss_mask[j:end_idx]
                
                outputs = model(input_ids=mini_input_ids, attention_mask=mini_attention_mask)
                # Get token-level log probs
                token_log_probs = self._get_token_log_probs(outputs.logits, mini_input_ids, mini_loss_mask)
                token_log_probs_list.append(token_log_probs.cpu())  # Move to CPU
            
            # [G, Seq-1]
            my_old_token_log_probs.append(torch.cat(token_log_probs_list, dim=0))
            
            # if (i + 1) % 10 == 0:
            #     print(f"[Rank {local_rank}] Processed {i + 1}/{len(my_indices)}")
        
        # 4. Gather results (All Gather)
        if world_size > 1:
            print(f"[Rank {local_rank}] Gathering results from all GPUs...")
            
            # Stack current rank results: [N_local, G, Seq-1]
            if len(my_old_token_log_probs) > 0:
                local_tensor = torch.stack(my_old_token_log_probs).to(device)
            else:
                # Empty tensor needs correct shape
                seq_minus_1 = self.max_length - 1
                local_tensor = torch.empty(0, self.group_size, seq_minus_1, device=device)

            # Handle inconsistent data sizes across cards (padding)
            local_size = torch.tensor([local_tensor.shape[0]], device=device)
            all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(all_sizes, local_size)
            max_size = max([s.item() for s in all_sizes])
            
            # Pad local_tensor to max_size
            pad_size = max_size - local_tensor.shape[0]
            if pad_size > 0:
                pad_tensor = torch.zeros((pad_size, *local_tensor.shape[1:]), dtype=local_tensor.dtype, device=device)
                padded_local_tensor = torch.cat([local_tensor, pad_tensor], dim=0)
            else:
                padded_local_tensor = local_tensor
                
            # Gather
            gathered_tensors = [torch.zeros_like(padded_local_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_tensors, padded_local_tensor)
            
            # 5. Fill back to processed_data
            for rank in range(world_size):
                count = all_sizes[rank].item()
                rank_tensor = gathered_tensors[rank][:count].cpu()  # [Count, G, Seq-1]
                
                rank_indices = [k for k in range(len(self.processed_data)) if k % world_size == rank]
                
                for k, data_idx in enumerate(rank_indices):
                    self.processed_data[data_idx]["old_token_log_probs"] = rank_tensor[k]
        else:
            # Single card assignment
            for k, idx in enumerate(my_indices):
                self.processed_data[idx]["old_token_log_probs"] = my_old_token_log_probs[k]

        # 6. Release memory
        print(f"[Rank {local_rank}] Offloading model and clearing cache...")
        model.cpu()
        torch.cuda.empty_cache()
    
    def _get_token_log_probs(self, logits, labels, loss_mask):
        """Calculate token-level log probs (masked)"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = loss_mask[..., 1:].contiguous().float()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        token_log_probs = -token_loss.view(labels.size(0), -1)  # [B, Seq-1]
        # 应用 mask，padding 位置为 0
        masked_token_log_probs = token_log_probs * shift_mask
        
        return masked_token_log_probs  # [B, Seq-1]

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def collate_fn(batch):
    """Stack data in batch: [B, G, Seq]"""
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "loss_mask": torch.stack([x["loss_mask"] for x in batch]),
        "rewards": torch.stack([x["rewards"] for x in batch]),
        "old_token_log_probs": torch.stack([x["old_token_log_probs"] for x in batch]),  # [B, G, Seq-1]
    }

# ===================
# $$
# \begin{gathered}
# \mathcal{J}_{G R P O}(\theta)=\mathbb{E}\left[q \sim P(Q),\left\{o_i\right\}_{i=1}^G \sim \pi_{\theta_{\text {old }}}(O \mid q)\right] \\
# \qquad \frac{1}{G} \sum_{i=1}^G\left(\min \left(\frac{\pi_\theta\left(o_i \mid q\right)}{\pi_{\theta_{\text {old }}}\left(o_i \mid q\right)} A_i, \operatorname{clip}\left(\frac{\pi_\theta\left(o_i \mid q\right)}{\pi_{\theta_{\text {old }}}\left(o_i \mid q\right)}, 1-\varepsilon, 1+\varepsilon\right) A_i\right)-\beta \mathbb{D}_{K L}\left(\pi_\theta \| \pi_{\text {ref }}\right)\right), \\
# \mathbb{D}_{K L}\left(\pi_\theta \| \pi_{\text {ref }}\right)=\frac{\pi_{\text {ref }}\left(o_i \mid q\right)}{\pi_\theta\left(o_i \mid q\right)}-\log \frac{\pi_{\text {ref }}\left(o_i \mid q\right)}{\pi_\theta\left(o_i \mid q\right)}-1,
# \end{gathered}
# $$
# $A_i=\frac{r_i-\operatorname{mean}\left(\left\{r_1, r_2, \cdots, r_G\right\}\right)}{\operatorname{std}\left(\left\{r_1, r_2, \cdots, r_G\right\}\right)}$.
# ===================


# ================= 2. Token-level GRPO Trainer =================
class GRPOTrainer(Trainer):
    def __init__(self, grpo_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grpo_cfg = grpo_config
        # Init log list
        self.training_logs = []
        os.makedirs(os.path.dirname(self.grpo_cfg.log_file), exist_ok=True)

    def log(self, logs, *args, **kwargs):
        """Override log method to save to JSON file"""
        super().log(logs, *args, **kwargs)
        # Add timestamp and global step
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            **logs
        }
        
        # self.training_logs.append(log_entry)
        
        # # 每 10 步或每个 epoch 保存一次
        # if self.state.global_step % 10 == 0 or logs.get("epoch"):
        #     self._save_logs()        

    def _save_logs(self):
        """Save logs to JSON"""
        try:
            with open(self.grpo_cfg.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_logs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save logs: {e}")
    
    def on_train_end(self):
        """Save final logs on train end"""
        super().on_train_end()
        self._save_logs()
        print(f"Training logs saved to {self.grpo_cfg.log_file}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Token-level GRPO Loss Calculation
        - Token-level PPO Clip
        - Token-level KL Penalty
        """
        b_size, g_size, seq_len = inputs["input_ids"].shape
        flatten_size = b_size * g_size
        seq_len_minus_1 = seq_len - 1
        
        input_ids = inputs["input_ids"].view(flatten_size, seq_len)
        attention_mask = inputs["attention_mask"].view(flatten_size, seq_len)
        loss_mask = inputs["loss_mask"].view(flatten_size, seq_len)
        rewards = inputs["rewards"]  # [B, G]
        old_token_log_probs = inputs["old_token_log_probs"].view(flatten_size, seq_len_minus_1)  # [B*G, Seq-1]
        
        # 1. Compute Advantages (Intra-group Standardization) - Still sequence-level
        mean_rewards = rewards.mean(dim=1, keepdim=True)
        std_rewards = rewards.std(dim=1, keepdim=True)
        advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)
        advantages = advantages.view(-1)  # [B*G]
        advantages = torch.clamp(advantages, -10.0, 10.0)
        
        # 2. Compute current policy Token-level Log Probs
        mini_batch_size = self.grpo_cfg.ppo_micro_batch_size_per_gpu
        current_token_log_probs_list = []
        shift_mask_list = []
        
        for i in range(0, flatten_size, mini_batch_size):
            end_idx = min(i + mini_batch_size, flatten_size)
            mini_input_ids = input_ids[i:end_idx]
            mini_attention_mask = attention_mask[i:end_idx]
            mini_loss_mask = loss_mask[i:end_idx]
            
            outputs = model(input_ids=mini_input_ids, attention_mask=mini_attention_mask)
            token_log_probs, shift_mask = self._get_token_log_probs(
                outputs.logits, mini_input_ids, mini_loss_mask
            )
            current_token_log_probs_list.append(token_log_probs)
            shift_mask_list.append(shift_mask)
        
        current_token_log_probs = torch.cat(current_token_log_probs_list, dim=0)  # [B*G, Seq-1]
        shift_mask = torch.cat(shift_mask_list, dim=0)  # [B*G, Seq-1]

        # 3. Token-level PPO Clipped Loss
        # log_ratio: [B*G, Seq-1]
        log_ratio = current_token_log_probs - old_token_log_probs
        ratio = torch.exp(log_ratio)
        
        # Expand advantages: [B*G] -> [B*G, 1]
        adv_unsqueezed = advantages.unsqueeze(1) 
        
        surr1 = ratio * adv_unsqueezed
        surr2 = torch.clamp(ratio, 1 - self.grpo_cfg.epsilon_low, 1 + self.grpo_cfg.epsilon_high) * adv_unsqueezed
        
        # Token-level loss: [B*G, Seq-1]
        token_policy_loss = -torch.min(surr1, surr2)
        
        # 4. Token-level KL Penalty (Approx KL: exp(log_diff) - log_diff - 1)
        # punishment for deviation: KL(new || old) approx
        
        # 5. Combine Loss and Average per Sequence
        # Calculate valid length per sequence
        valid_len_per_seq = shift_mask.sum(dim=1) # [B*G]
        
        # Compute policy loss and kl loss per sequence
        masked_policy_loss = token_policy_loss * shift_mask
        seq_policy_loss = masked_policy_loss.sum(dim=1) / (valid_len_per_seq + 1e-8)
        
        # Compute total loss for each sequence
        # per_seq_loss = seq_policy_loss + seq_kl_loss # [B*G]
        per_seq_loss = seq_policy_loss
        # Final Loss is mean of all sequence losses
        loss = per_seq_loss.mean()

        return (loss, None) if return_outputs else loss
    
    def _get_token_log_probs(self, logits, labels, loss_mask):
        """Calculate token-level log probs, return shift_mask as well"""
        # Avoid contiguous on full logits (B, Seq, Vocab) to prevent OOM
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        shift_mask = loss_mask[..., 1:].contiguous().float()
        
        batch_size, seq_len, vocab_size = shift_logits.shape
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_log_probs_list = []
        chunk_size = 512 
        
        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            
            # Only copy chunk to contiguous memory
            chunk_logits = shift_logits[:, i:end, :].contiguous()
            chunk_labels = shift_labels[:, i:end].contiguous()
            
            chunk_loss = loss_fct(
                chunk_logits.view(-1, vocab_size),
                chunk_labels.view(-1)
            )
            token_log_probs_list.append(chunk_loss.view(batch_size, -1))
            
        token_log_probs = -torch.cat(token_log_probs_list, dim=1)
        masked_token_log_probs = token_log_probs * shift_mask
        
        return masked_token_log_probs, shift_mask
    
    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )


# ================= 3. Main Program =================
def main():
    # 1. Init default config
    cfg = GRPOConfig()
    
    # 2. Parse config path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    # 3. Load and override config
    if args.config:
        print(f"Loading config from {args.config}")
        with open(args.config, "r") as f:
            json_config = json.load(f)
            
            # Map JSON config to cfg object
            if "model_path" in json_config:
                cfg.model_name = json_config["model_path"]
            if "training_data_path" in json_config:
                cfg.dataset_path = json_config["training_data_path"]
            if "output_dir" in json_config:
                cfg.output_dir = json_config["output_dir"]
            # New: Support ppo_micro_batch_size_per_gpu
            if "ppo_micro_batch_size_per_gpu" in json_config:
                cfg.ppo_micro_batch_size_per_gpu = json_config["ppo_micro_batch_size_per_gpu"]
                print(f"Using ppo_micro_batch_size_per_gpu={cfg.ppo_micro_batch_size_per_gpu}")
            
            # Update log file path
            os.makedirs(cfg.output_dir, exist_ok=True)
            cfg.log_file = os.path.join(cfg.output_dir, f"training_log.json")

    print(f"Training Config: Model={cfg.model_name}, Data={cfg.dataset_path}, Output={cfg.output_dir}")

    print("Loading data...")
    data = read_json(cfg.dataset_path)
    random.shuffle(data)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' 
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
    # Create dataset and precompute token-level old_log_probs
    print("Creating dataset and pre-computing token-level old log probs...")
    dataset = GRPODataset(
        data, tokenizer, 
        max_length=cfg.max_length, 
        group_size=cfg.group_size,
        model=model,
        mini_batch_size=cfg.log_prob_micro_batch_size_per_gpu * cfg.group_size,
    )
    
    # Enable gradient checkpointing after pre-computation
    model.gradient_checkpointing_enable()
    
    if cfg.use_lora:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        deepspeed="ds_config_grpo.json",
    )
    
    trainer = GRPOTrainer(
        grpo_config=cfg,
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    trainer.save_model()
    print(f"Model saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()