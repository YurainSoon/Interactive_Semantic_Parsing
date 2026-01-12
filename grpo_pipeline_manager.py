"""
GRPO Training Pipeline Manager
Coordinates the full process: Rollout (vLLM) -> Reward (Scoring) -> Training (HuggingFace)
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from yurain_utils import *

class GRPOPipelineManager:
    """GRPO Training Pipeline Manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Base path config (default to current dir)
        self.base_dir = Path(config.get("base_dir", "."))
        self.results_dir = Path(config.get("results_dir", "./results/cwq"))
        self.models_dir = Path(config.get("models_dir", "./grpo_model"))
        
        # Mode config: node (k-beam) or trajectory
        self.mode = config.get("mode", "node")
        
        # Script paths
        if self.mode == "trajectory":
            self.rollout_script = self.base_dir / "infer_reasoning.py"
            self.scoring_script = self.base_dir / "trajectory_scorer.py"
        else:
            self.rollout_script = self.base_dir / "infer_reasoning_k_beam.py"
            self.scoring_script = self.base_dir / "k_beam_score.py"
            
        self.training_script = self.base_dir / "grpo.py"
        
        # Training config
        self.initial_model = config.get("initial_model", "./models/base_model")
        self.dataset_path = config.get("dataset_path", "./datasets/train.json")
        
        # Batch config
        self.batch_size = config.get("batch_size", 128)
        self.total_samples = config.get("total_samples", None)  # If None, use full dataset
        self.k_samples = config.get("k_samples", 16)
        self.max_interactions = config.get("max_interactions", 6)
        
        # Epoch config
        self.total_epochs = config.get("total_epochs", 1)
        self.start_epoch = config.get("start_epoch", 0)
        
        # GPU config
        self.rollout_gpu = config.get("rollout_gpu", "0")
        self.training_gpu = config.get("training_gpu", "0,1,2,3")
        
        # Cleanup config
        self.keep_last_n_checkpoints = config.get("keep_last_n_checkpoints", 2)
        self.save_every_n_steps = config.get("save_every_n_steps", 0)  # 0 means not enabled
        self.clean_intermediate_files = config.get("clean_intermediate_files", True)
        
        # Logging
        self.log_dir = self.base_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.current_model_path = self.initial_model
        self.completed_epochs = []
        
    def log(self, message: str, level: str = "INFO"):
        """Record log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        
        log_file = self.log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, "a") as f:
            f.write(log_message + "\n")
    
    def get_epoch_dir(self, epoch: int) -> Path:
        """Get results dir for a specific epoch"""
        return self.results_dir / f"epoch_{epoch}"
    
    def run_command(self, cmd: list, description: str, env: Optional[Dict] = None) -> bool:
        """Run command and wait for completion"""
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(cmd)}")
        
        try:
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                env=process_env,
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                self.log(f"✓ {description} completed successfully")
                return True
            else:
                self.log(f"✗ {description} failed with return code {result.returncode}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"✗ {description} failed with exception: {e}", "ERROR")
            return False
    
    def run_rollout(self, epoch: int, start_batch: int, end_batch: int) -> bool:
        """Run Rollout Phase (vLLM inference)"""
        epoch_dir = self.get_epoch_dir(epoch)
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = f"epoch_{epoch}/{start_batch}_{end_batch}_{self.k_samples}"
        
        result_subdir = self.results_dir / save_path

        # Check if output exists
        if self.mode == "trajectory":
            output_check_path = result_subdir / "data_states.json"
            if output_check_path.exists():
                self.log(f"Rollout output found at {output_check_path}, skipping rollout.")
                return True
        else:
            trajectory_tree_path = result_subdir / "k_sampling_trajectory_tree.json"
            sample_summaries_path = result_subdir / "k_sampling_sample_summaries.json"
            if trajectory_tree_path.exists() and sample_summaries_path.exists():
                self.log(f"Rollout outputs found in {result_subdir}, skipping rollout.")
                return True
        print(f"results_dir: {self.results_dir}")
        
        # Generate temporary config
        rollout_config = {
            "model_path": self.current_model_path,
            "dataset_path": self.dataset_path,
            "start_epoch": start_batch,
            "end_epoch": end_batch,
            "batch_size": self.batch_size,
            "k_samples": self.k_samples,
            "max_interactions": self.max_interactions,
            "save_path": save_path,
            "results_dir": str(self.results_dir),
        }
        
        config_path = epoch_dir / "rollout_config.json"
        with open(config_path, "w") as f:
            json.dump(rollout_config, f, indent=2)
        
        # Run inference script
        cmd = [
            "python", str(self.rollout_script),
            "--config", str(config_path)
        ]
        
        env = {"CUDA_VISIBLE_DEVICES": self.rollout_gpu}
        
        return self.run_command(cmd, f"Rollout (Epoch {epoch}, Batch {start_batch}-{end_batch})", env)
    
    def run_scoring(self, epoch: int, start_batch: int, end_batch: int) -> bool:
        """Run Scoring Phase"""
        save_path = f"epoch_{epoch}/{start_batch}_{end_batch}_{self.k_samples}"
        result_subdir = self.results_dir / save_path

        distance_api = self.config.get("distance_api", "http://localhost:5501/entity_distance")
        
        if self.mode == "trajectory":
            input_path = result_subdir / "data_states.json"
            output_path = result_subdir / "scored_training_data.json"

            # Check if output exists
            if output_path.exists():
                self.log(f"Scoring output found at {output_path}, skipping scoring.")
                return True

            if not input_path.exists():
                self.log(f"Input file not found: {input_path}", "ERROR")
                return False
                
            cmd = [
                "python", str(self.scoring_script),
                "--input_path", str(input_path),
                "--output_path", str(output_path),
                "--distance_api", distance_api,
                "--max_distance", "3"
            ]
        else:
            trajectory_tree_path = result_subdir / "k_sampling_trajectory_tree.json"
            sample_summaries_path = result_subdir / "k_sampling_sample_summaries.json"
            output_path = result_subdir / "k_sampling_scored_training_data.json"
            
            # Check if output exists
            if output_path.exists():
                self.log(f"Scoring output found at {output_path}, skipping scoring.")
                return True

            # Check if input exists
            if not trajectory_tree_path.exists() or not sample_summaries_path.exists():
                self.log(f"Input files not found in {result_subdir}", "ERROR")
                return False
            
            cmd = [
                "python", str(self.scoring_script),
                "--trajectory_tree_path", str(trajectory_tree_path),
                "--sample_summaries_path", str(sample_summaries_path),
                "--output_path", str(output_path),
                "--distance_api", distance_api, # Pass API URL as Arg
            ]
        
        return self.run_command(cmd, f"Scoring (Epoch {epoch}, Batch {start_batch}-{end_batch})")
    
    def run_training(self, epoch: int, training_data_path: list, start_batch: int, end_batch: int) -> bool:
        """Run Training Phase (HuggingFace GRPO), supports auto-downgrade retry"""
        epoch_dir = self.get_epoch_dir(epoch)
        epoch_dir.mkdir(parents=True, exist_ok=True)
        training_data = read_json(training_data_path)
        if not training_data:
            self.log(f"No training data found for epoch {epoch}, batch {start_batch}-{end_batch}", "ERROR")
            return False
        
        # Set output model path (include batch info)
        new_model_path = self.models_dir / f"epoch_{epoch}/batch_{start_batch}_{end_batch}"
        
        # Try different ppo_micro_batch_size_per_gpu values
        ppo_batch_sizes = [6, 4]
        
        for attempt, ppo_batch_size in enumerate(ppo_batch_sizes, 1):
            self.log(f"Training attempt {attempt}/{len(ppo_batch_sizes)} with ppo_micro_batch_size_per_gpu={ppo_batch_size}")
            
            # Generate training config
            training_config = {
                "model_path": self.current_model_path,
                "training_data_path": str(training_data_path),
                "output_dir": str(new_model_path),
                "epoch": epoch,
                "ppo_micro_batch_size_per_gpu": ppo_batch_size,
            }
            
            config_path = epoch_dir / f"training_config_batch_{start_batch}_{end_batch}_attempt{attempt}.json"
            with open(config_path, "w") as f:
                json.dump(training_config, f, indent=2)
            
            # Change command to deepspeed
            cmd = [
                "deepspeed", 
                "--num_gpus=4",
                str(self.training_script),
                "--config", str(config_path)
            ]
            
            env = {"CUDA_VISIBLE_DEVICES": self.training_gpu}
            
            success = self.run_command(cmd, f"Training (Epoch {epoch}, Batch {start_batch}-{end_batch}, PPO Batch={ppo_batch_size})", env)
            
            if success:
                # Update current model path for next batch
                self.current_model_path = str(new_model_path)
                self.log(f"Updated current model to: {self.current_model_path}")
                self.log(f"Training succeeded with ppo_micro_batch_size_per_gpu={ppo_batch_size}")
                return True
            else:
                self.log(f"Training failed with ppo_micro_batch_size_per_gpu={ppo_batch_size}", "WARNING")
                
                # If not last attempt, continue
                if attempt < len(ppo_batch_sizes):
                    self.log(f"Retrying with smaller ppo_micro_batch_size_per_gpu...", "INFO")
                else:
                    self.log(f"All training attempts failed for batch {start_batch}-{end_batch}", "ERROR")
        
        return False
    
    def cleanup_old_checkpoints(self):
        """Cleanup old model checkpoints"""
        # If policy not configured, return
        if self.keep_last_n_checkpoints <= 0 and self.save_every_n_steps <= 0:
            return
        
        # Parse dir name for sorting: epoch_X_batch_Y_Z
        def parse_checkpoint_name(name):
            try:
                parts = name.split("_")
                # Format: epoch, {epoch_num}, batch, {start}, {end}
                if len(parts) >= 5 and parts[0] == "epoch" and parts[2] == "batch":
                    return int(parts[1]), int(parts[4]) # epoch, end_batch
                # Compatible with old format epoch_{epoch}
                elif len(parts) == 2 and parts[0] == "epoch":
                    return int(parts[1]), 0
                return -1, -1
            except:
                return -1, -1

        # Get all checkpoints and sort
        checkpoints = [d for d in self.models_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
        checkpoints = sorted(checkpoints, key=lambda x: parse_checkpoint_name(x.name))
        
        if not checkpoints:
            return

        # Set to store paths to keep
        checkpoints_to_keep = set()

        # Strategy 1: Always keep last N models
        n_to_keep = max(1, self.keep_last_n_checkpoints)
        for ckpt in checkpoints[-n_to_keep:]:
            checkpoints_to_keep.add(ckpt)

        # Strategy 2: Keep every save_every_n_steps (0, 8, 16...)
        if self.save_every_n_steps > 0:
            for ckpt in checkpoints:
                try:
                    parts = ckpt.name.split("_")
                    # Format: epoch_{epoch}_batch_{start}_{end}
                    if len(parts) >= 5 and parts[2] == "batch":
                        start_batch = int(parts[3])
                        if start_batch % self.save_every_n_steps == 0:
                            checkpoints_to_keep.add(ckpt)
                            self.log(f"Keeping checkpoint at step {start_batch}: {ckpt.name}")
                except Exception:
                    pass

        # Cleanup: Delete checkpoints not in keep list
        for ckpt in checkpoints:
            if ckpt not in checkpoints_to_keep:
                self.log(f"Cleaning up old checkpoint: {ckpt.name}")
                shutil.rmtree(ckpt)
    
    def cleanup_intermediate_files(self, epoch: int):
        pass
    
    def get_dataset_size(self) -> int:
        """Get dataset size"""
        if self.total_samples:
            return self.total_samples
        
        try:
            with open(self.dataset_path, "r") as f:
                data = json.load(f)
                return len(data)
        except Exception as e:
            self.log(f"Failed to load dataset: {e}", "ERROR")
            return 0
    
    def run_epoch(self, epoch: int) -> bool:
        """Run a full epoch"""
        self.log(f"=" * 60)
        self.log(f"Starting Epoch {epoch}")
        self.log(f"=" * 60)
        
        dataset_size = self.get_dataset_size()
        if dataset_size == 0:
            self.log("Dataset is empty or failed to load", "ERROR")
            return False
        
        num_batches = (dataset_size + self.batch_size - 1) // self.batch_size
        self.log(f"Dataset size: {dataset_size}, Batch size: {self.batch_size}, Num batches: {num_batches}")
        
        # Batch processing
        for batch_idx in range(num_batches):
            # start_batch and end_batch match start_epoch/end_epoch in infer_reasoning_k_beam.py (batch index)
            start_batch_idx = batch_idx
            end_batch_idx = batch_idx + 1
            
            # Calculate real sample index for logging and model naming
            real_start_idx = batch_idx * self.batch_size
            real_end_idx = min((batch_idx + 1) * self.batch_size, dataset_size)
            
            self.log(f"-" * 40)
            self.log(f"Processing Batch {batch_idx + 1}/{num_batches} (Samples {real_start_idx}-{real_end_idx})")
            
            # Step 1: Rollout
            if not self.run_rollout(epoch, start_batch_idx, end_batch_idx):
                self.log(f"Rollout failed for batch {batch_idx}", "ERROR")
                return False # 如果 rollout 失败，中断流程
            
            # Step 2: Scoring
            if not self.run_scoring(epoch, start_batch_idx, end_batch_idx):
                self.log(f"Scoring failed for batch {batch_idx}", "ERROR")
                return False
            
            # 收集当前批次的训练数据路径
            save_path = f"epoch_{epoch}/{start_batch_idx}_{end_batch_idx}_{self.k_samples}"
            
            if self.mode == "trajectory":
                training_data_path = self.results_dir / save_path / "scored_training_data.json"
            else:
                training_data_path = self.results_dir / save_path / "k_sampling_scored_training_data.json"
            
            if not training_data_path.exists():
                self.log(f"Training data not found for batch {batch_idx}: {training_data_path}", "ERROR")
                return False

            # Step 3: Training (立即训练)
            # 使用实际样本索引作为模型名称的一部分
            if not self.run_training(epoch, str(training_data_path), real_start_idx, real_end_idx):
                self.log(f"Training failed for batch {batch_idx}", "ERROR")
                return False
            
            # Step 4: Cleanup (Only clear old models)
            self.cleanup_old_checkpoints()
        
        return True
    
    def run(self):
        """Run full training pipeline"""
        self.log("=" * 60)
        self.log("GRPO Pipeline Manager Started")
        self.log(f"Initial model: {self.initial_model}")
        self.log(f"Total epochs: {self.total_epochs}")
        self.log(f"Start epoch: {self.start_epoch}")
        self.log("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.start_epoch + self.total_epochs):
            epoch_start = time.time()
            
            success = self.run_epoch(epoch)
            
            epoch_duration = time.time() - epoch_start
            self.log(f"Epoch {epoch} duration: {epoch_duration / 60:.2f} minutes")
            
            if not success:
                self.log(f"Epoch {epoch} failed, stopping pipeline", "ERROR")
                break
        
        total_duration = time.time() - start_time
        self.log("=" * 60)
        self.log("GRPO Pipeline Manager Finished")
        self.log(f"Completed epochs: {self.completed_epochs}")
        self.log(f"Final model: {self.current_model_path}")
        self.log(f"Total duration: {total_duration / 3600:.2f} hours")
        self.log("=" * 60)
        
        return len(self.completed_epochs) > 0


def main():
    parser = argparse.ArgumentParser(description="GRPO Training Pipeline Manager")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--initial_model", type=str, help="Initial model path")
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    parser.add_argument("--total_epochs", type=int, help="Total epochs to train")
    parser.add_argument("--start_epoch", type=int, help="Start epoch")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--k_samples", type=int, help="K samples for beam search")
    parser.add_argument("--rollout_gpu", type=str, help="GPU for rollout")
    parser.add_argument("--training_gpu", type=str, help="GPU for training")
    parser.add_argument("--mode", type=str, help="Mode: 'node' or 'trajectory'")
    
    args = parser.parse_args()
    
    # Load Config
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override with CLI args
    if args.mode:
        config["mode"] = args.mode
    if args.initial_model:
        config["initial_model"] = args.initial_model
    if args.dataset_path:
        config["dataset_path"] = args.dataset_path
    if args.total_epochs:
        config["total_epochs"] = args.total_epochs
    if args.start_epoch:
        config["start_epoch"] = args.start_epoch
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.k_samples:
        config["k_samples"] = args.k_samples
    if args.rollout_gpu:
        config["rollout_gpu"] = args.rollout_gpu
    if args.training_gpu:
        config["training_gpu"] = args.training_gpu
    
    # Run Manager
    manager = GRPOPipelineManager(config)
    success = manager.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()