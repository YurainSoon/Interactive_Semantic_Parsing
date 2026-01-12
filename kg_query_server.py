from flask import Flask, request, jsonify
import re
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.parse import urlparse
import numpy as np
from openai import OpenAI
import string
import hashlib
import pickle
import os
import time
from threading import Lock
from collections import defaultdict, deque
from datetime import datetime
import logging
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from kg_function import *


# ============ New: Logging Configuration ============
def setup_logging():
    """Configure logging system"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('kg_query_service.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============ Configuration ============
SPARQL_ENDPOINT = os.getenv("SPARQL_ENDPOINT", "http://localhost:8890/sparql")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "retrieval_relation")
EMBEDDING_PORTS = [int(p) for p in os.getenv("EMBEDDING_PORTS", "8000,8001,8002,8003").split(",")]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen.Qwen3-Embedding-8B")

# ============ New: Service Metrics Class ============
class ServiceMetrics:
    """Service runtime metrics statistics"""
    def __init__(self, max_history=1000):
        self.lock = Lock()
        self.max_history = max_history
        
        # Request stats
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Query type stats
        self.query_type_counts = defaultdict(int)
        
        # Cache stats
        self.cache_hits = 0
        self.cache_misses = 0
        
        # SPARQL query stats
        self.sparql_queries = 0
        self.node_queries = 0
        
        # Response time stats
        self.response_times = deque(maxlen=max_history)
        
        # Error stats
        self.errors = deque(maxlen=100)
        
        # Relation retrieval stats
        self.relation_retrieval_stats = {
            'total_relations_found': 0,
            'total_after_filter': 0,
            'filter_applied_count': 0
        }
        
        # ============ New: Distance Query Stats ============
        self.distance_query_stats = {
            'total_distance_queries': 0,
            'total_entity_pairs': 0,
            'cache_hit_pairs': 0,
            'actual_sparql_pairs': 0,
            'distance_distribution': defaultdict(int),
            'set_query_count': 0,
            'avg_set_size_a': 0.0,
            'avg_set_size_b': 0.0,
            'early_stop_triggered': 0,
        }
        
        # Start time
        self.start_time = datetime.now()
        
        # Recent request history
        self.recent_requests = deque(maxlen=100)
    
    def record_request(self, query_type, success=True, response_time=0, error_msg=None):
        """Record request"""
        with self.lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
                if error_msg:
                    self.errors.append({
                        'timestamp': datetime.now().isoformat(),
                        'error': error_msg,
                        'query_type': query_type
                    })
            
            self.query_type_counts[query_type] += 1
            self.response_times.append(response_time)
            
            self.recent_requests.append({
                'timestamp': datetime.now().isoformat(),
                'type': query_type,
                'success': success,
                'response_time': response_time
            })
    
    def record_cache(self, hit=True):
        """Record cache usage"""
        with self.lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def record_query_execution(self, query_type):
        """Record query execution"""
        with self.lock:
            if query_type == 'sparql':
                self.sparql_queries += 1
            elif query_type == 'node':
                self.node_queries += 1
    
    def record_relation_retrieval(self, original_count, filtered_count, filter_applied):
        """Record relation retrieval stats"""
        with self.lock:
            self.relation_retrieval_stats['total_relations_found'] += original_count
            self.relation_retrieval_stats['total_after_filter'] += filtered_count
            if filter_applied:
                self.relation_retrieval_stats['filter_applied_count'] += 1
    
    def record_distance_query(self, set_a_size, set_b_size, overall_distance, 
                             cache_hits, actual_queries, early_stopped=False):
        """
        Record distance query statistics
        
        Args:
            set_a_size: Size of Set A
            set_b_size: Size of Set B
            overall_distance: Minimum overall distance
            cache_hits: Number of cache hits
            actual_queries: Number of actual SPARQL queries executed
            early_stopped: Whether early stopping was triggered
        """
        with self.lock:
            self.distance_query_stats['total_distance_queries'] += 1
            self.distance_query_stats['set_query_count'] += 1
            
            total_pairs = set_a_size * set_b_size if set_b_size > 0 else 0
            self.distance_query_stats['total_entity_pairs'] += total_pairs
            self.distance_query_stats['cache_hit_pairs'] += cache_hits
            self.distance_query_stats['actual_sparql_pairs'] += actual_queries
            
            # Update distance distribution
            self.distance_query_stats['distance_distribution'][overall_distance] += 1
            
            # Update average set sizes
            total_queries = self.distance_query_stats['set_query_count']
            self.distance_query_stats['avg_set_size_a'] = (
                (self.distance_query_stats['avg_set_size_a'] * (total_queries - 1) + set_a_size) 
                / total_queries
            )
            self.distance_query_stats['avg_set_size_b'] = (
                (self.distance_query_stats['avg_set_size_b'] * (total_queries - 1) + set_b_size) 
                / total_queries
            )
            
            if early_stopped:
                self.distance_query_stats['early_stop_triggered'] += 1

    def get_stats(self):
        """Get statistics data"""
        with self.lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            max_response_time = max(self.response_times) if self.response_times else 0
            min_response_time = min(self.response_times) if self.response_times else 0
            
            cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0
            
            success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
            
            avg_filter_reduction = 0
            if self.relation_retrieval_stats['filter_applied_count'] > 0:
                avg_filter_reduction = (
                    (self.relation_retrieval_stats['total_relations_found'] - 
                     self.relation_retrieval_stats['total_after_filter']) / 
                    self.relation_retrieval_stats['total_relations_found'] * 100
                ) if self.relation_retrieval_stats['total_relations_found'] > 0 else 0
            
            # ============ New: Calculate distance query cache hit rate ============
            distance_cache_hit_rate = 0
            if self.distance_query_stats['total_entity_pairs'] > 0:
                distance_cache_hit_rate = (
                    self.distance_query_stats['cache_hit_pairs'] / 
                    self.distance_query_stats['total_entity_pairs'] * 100
                )
            
            # Calculate early stop efficiency
            early_stop_rate = 0
            if self.distance_query_stats['set_query_count'] > 0:
                early_stop_rate = (
                    self.distance_query_stats['early_stop_triggered'] / 
                    self.distance_query_stats['set_query_count'] * 100
                )
            
            return {
                'uptime_seconds': uptime,
                'uptime_formatted': self._format_uptime(uptime),
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': f"{success_rate:.2f}%",
                'query_type_distribution': dict(self.query_type_counts),
                'cache_stats': {
                    'hits': self.cache_hits,
                    'misses': self.cache_misses,
                    'hit_rate': f"{cache_hit_rate:.2f}%"
                },
                'query_execution': {
                    'sparql_queries': self.sparql_queries,
                    'node_queries': self.node_queries
                },
                'response_time': {
                    'average_ms': f"{avg_response_time * 1000:.2f}",
                    'max_ms': f"{max_response_time * 1000:.2f}",
                    'min_ms': f"{min_response_time * 1000:.2f}"
                },
                'relation_retrieval': {
                    'total_found': self.relation_retrieval_stats['total_relations_found'],
                    'total_after_filter': self.relation_retrieval_stats['total_after_filter'],
                    'filter_applied_count': self.relation_retrieval_stats['filter_applied_count'],
                    'avg_filter_reduction': f"{avg_filter_reduction:.2f}%"
                },
                # ============ New: Distance Query Stats ============
                'distance_query_stats': {
                    'total_queries': self.distance_query_stats['total_distance_queries'],
                    'set_queries': self.distance_query_stats['set_query_count'],
                    'total_entity_pairs': self.distance_query_stats['total_entity_pairs'],
                    'cache_hit_pairs': self.distance_query_stats['cache_hit_pairs'],
                    'actual_sparql_pairs': self.distance_query_stats['actual_sparql_pairs'],
                    'cache_hit_rate': f"{distance_cache_hit_rate:.2f}%",
                    'avg_set_size_a': f"{self.distance_query_stats['avg_set_size_a']:.1f}",
                    'avg_set_size_b': f"{self.distance_query_stats['avg_set_size_b']:.1f}",
                    'early_stop_rate': f"{early_stop_rate:.2f}%",
                    'distance_distribution': dict(self.distance_query_stats['distance_distribution'])
                },
                'recent_errors': list(self.errors)[-10:],
                'recent_requests': list(self.recent_requests)[-20:]
            }
    
    def _format_uptime(self, seconds):
        """Format uptime"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{days}d {hours}h {minutes}m {secs}s"

# ============ Modified Cache Class to support metrics ============
class QueryRelationCache:
    def __init__(self, cache_file="query_relation_cache.pkl", auto_unload_seconds=300, metrics=None):
        self.cache_file = cache_file
        self.cache = None
        self.auto_unload_seconds = auto_unload_seconds
        self.last_access_time = None
        self.lock = Lock()
        self.metrics = metrics  # New: Metrics object
    
    def _load_cache(self):
        """Load cache file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # logger.info(f"Cache loaded successfully with {len(cache_data)} entries")
                    return cache_data
            except (pickle.UnpicklingError, EOFError, IOError) as e:
                logger.warning(f"Failed to load cache file: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        if self.cache is not None:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
                # logger.info(f"Cache saved with {len(self.cache)} entries")

    def _ensure_loaded(self):
        """Ensure cache is loaded"""
        with self.lock:
            if self.cache is None:
                self.cache = self._load_cache()
            self.last_access_time = time.time()

    def _check_and_unload(self):
        """Check if cache needs to be unloaded"""
        with self.lock:
            if (self.cache is not None and 
                self.last_access_time is not None and
                time.time() - self.last_access_time > self.auto_unload_seconds):
                self._save_cache()
                # logger.info("Cache unloaded due to inactivity")
                self.cache = None
                self.last_access_time = None

    def _get_cache_key(self, node="", previous=""):
        """Generate cache key using original parameters"""
        key_components = [str(node), str(previous)]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_or_compute(self, node="", previous="", compute_func=None, forced_use_cache=True):
        """Get result from cache or compute new result"""
        self._check_and_unload()
        self._ensure_loaded()
        
        cache_key = self._get_cache_key(node=node, previous=previous)
        
        # Query cache
        if cache_key in self.cache and forced_use_cache:
            if self.metrics:
                self.metrics.record_cache(hit=True)
            logger.debug(f"Cache hit for node={node}")
            return self.cache[cache_key]
        
        # Cache miss
        if self.metrics:
            self.metrics.record_cache(hit=False)
        logger.debug(f"Cache miss for node={node}")
        
        # Compute
        if compute_func is not None:
            result = compute_func()
            with self.lock:
                if result:
                    self.cache[cache_key] = result
                self.last_access_time = time.time()
                self._save_cache()
            return result
        
        return None

    def manual_save(self):
        """Manually save cache"""
        self._save_cache()

    def manual_unload(self):
        """Manually unload cache"""
        with self.lock:
            self._save_cache()
            self.cache = None
            self.last_access_time = None

class DistanceCache:
    """Cache for distances between entity pairs"""
    def __init__(self, cache_file="distance_cache.pkl", auto_unload_seconds=300, metrics=None):
        self.cache_file = cache_file
        self.cache = None
        self.auto_unload_seconds = auto_unload_seconds
        self.last_access_time = None
        self.lock = Lock()
        self.metrics = metrics
    
    def _load_cache(self):
        """Load cache file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    # logger.info(f"Distance cache loaded successfully with {len(cache_data)} entries")
                    return cache_data
            except (pickle.UnpicklingError, EOFError, IOError) as e:
                logger.warning(f"Failed to load distance cache file: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        if self.cache is not None:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
                # logger.info(f"Distance cache saved with {len(self.cache)} entries")

    def _ensure_loaded(self):
        """Ensure cache is loaded"""
        with self.lock:
            if self.cache is None:
                self.cache = self._load_cache()
            self.last_access_time = time.time()

    def _check_and_unload(self):
        """Check if cache needs to be unloaded"""
        with self.lock:
            if (self.cache is not None and 
                self.last_access_time is not None and
                time.time() - self.last_access_time > self.auto_unload_seconds):
                self._save_cache()
                # logger.info("Distance cache unloaded due to inactivity")
                self.cache = None
                self.last_access_time = None

    def _get_cache_key(self, entity_a, entity_b):
        """Generate cache key (ensures a,b and b,a get the same key)"""
        sorted_pair = tuple(sorted([str(entity_a), str(entity_b)]))
        return hashlib.md5("|".join(sorted_pair).encode()).hexdigest()

    def get(self, entity_a, entity_b):
        """Get cached distance"""
        self._check_and_unload()
        self._ensure_loaded()
        
        cache_key = self._get_cache_key(entity_a, entity_b)
        
        if cache_key in self.cache:
            if self.metrics:
                self.metrics.record_cache(hit=True)
            # logger.debug(f"Distance cache hit for ({entity_a}, {entity_b})")
            return self.cache[cache_key]
        
        if self.metrics:
            self.metrics.record_cache(hit=False)
        return None
    
    def set(self, entity_a, entity_b, distance):
        """Set cached distance"""
        self._ensure_loaded()
        
        cache_key = self._get_cache_key(entity_a, entity_b)
        
        with self.lock:
            self.cache[cache_key] = distance
            self.last_access_time = time.time()
            self._save_cache()
    
    def manual_save(self):
        """Manually save cache"""
        self._save_cache()

    def manual_unload(self):
        """Manually unload cache"""
        with self.lock:
            self._save_cache()
            self.cache = None
            self.last_access_time = None

# ============ New: Embedding Client Pool (Sequential Version) ============
class EmbeddingClientPool:
    """OpenAI embedding client pool - Ensures return order"""
    def __init__(self, base_ports=None, 
                 base_url_template="http://localhost:{}/v1",
                 api_key="EMPTY",
                 max_texts_per_request=500):
        
        if base_ports is None:
            base_ports = EMBEDDING_PORTS
            
        self.clients = []
        for port in base_ports:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url_template.format(port)
            )
            self.clients.append(client)
        
        self.current_index = 0
        self.lock = Lock()
        self.num_clients = len(self.clients)
        self.max_texts_per_request = max_texts_per_request  # New: Save limit parameter
        # logger.info(f"Initialized EmbeddingClientPool with {self.num_clients} clients on ports {base_ports}, max_texts_per_request={max_texts_per_request}")
    
    def get_client(self):
        """Round-robin get client"""
        with self.lock:
            client = self.clients[self.current_index]
            self.current_index = (self.current_index + 1) % self.num_clients
            return client
    
    def batch_embed_parallel(self, texts, task_description="", model="Qwen.Qwen3-Embedding-8B"):
        """
        Batch get embeddings in parallel, strictly preserving order
        
        Strategy:
        1. Split texts into small chunks (each chunk <= max_texts_per_request)
        2. Assign chunks to available clients
        3. Reorder results by index after parallel requests
        """
        if not texts:
            return np.array([])
        
        num_texts = len(texts)
        
        # New: Split all texts into small chunks
        chunks_with_indices = []
        for i in range(0, num_texts, self.max_texts_per_request):
            end_idx = min(i + self.max_texts_per_request, num_texts)
            chunk_texts = texts[i:end_idx]
            chunk_indices = list(range(i, end_idx))
            chunks_with_indices.append((chunk_texts, chunk_indices))
        
        # Parallel request - using ThreadPool
        results_dict = {}  # {original_index: embedding}
        # logger.info(f"Starting parallel embedding requests...")
        # logger.info(f"Total texts: {num_texts}, Max per request: {self.max_texts_per_request}, Number of batches: {len(chunks_with_indices)}")
        
        with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
            # Submit all tasks
            future_to_chunk = {}
            for chunk_idx, (chunk_texts, chunk_indices) in enumerate(chunks_with_indices):
                if not chunk_texts:
                    continue
                
                # Use round-robin client
                client = self.get_client()
                # print(f"Submitting chunk {chunk_idx+1}/{len(chunks_with_indices)} with {len(chunk_texts)} texts to client {client.api_key}")
                future = executor.submit(
                    self._get_embeddings_single,
                    client,
                    chunk_texts,
                    model
                )
                future_to_chunk[future] = chunk_indices
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_indices = future_to_chunk[future]
                try:
                    embeddings = future.result()
                    # Map embeddings to original index
                    for idx, emb in zip(chunk_indices, embeddings):
                        results_dict[idx] = emb
                except Exception as e:
                    logger.error(f"Embedding request failed for indices {chunk_indices}: {e}")
                    # Fill failed with zero vector
                    for idx in chunk_indices:
                        results_dict[idx] = None
        
        # Reorder by original index
        ordered_embeddings = []
        for i in range(num_texts):
            if i in results_dict and results_dict[i] is not None:
                ordered_embeddings.append(results_dict[i])
            else:
                # If index is missing, fill with zero vector (infer dimension from first success)
                if ordered_embeddings:
                    dim = len(ordered_embeddings[0])
                else:
                    dim = 1024  # Default dimension
                logger.warning(f"Missing embedding at index {i}, using zero vector")
                ordered_embeddings.append([0.0] * dim)
        
        return np.array(ordered_embeddings)
    
    def _get_embeddings_single(self, client, texts, model):
        """Get embedding from single client"""
        response = client.embeddings.create(
            input=texts,
            model=model
        )
        return np.array([item.embedding for item in response.data])



app = Flask(__name__)

class KGQueryService:
    def __init__(self, metrics=None):
        # SPARQL endpoint config
        self.sparql_endpoint = SPARQL_ENDPOINT
        
        # ============ Modified: Use pool instead of single client ============
        # Init Embedding Client Pool
        self.embedding_pool = EmbeddingClientPool(
            base_ports=EMBEDDING_PORTS,
            base_url_template="http://localhost:{}/v1",
            api_key=EMBEDDING_API_KEY
        )
        
        # Keep original client for compatibility (if used elsewhere)
        self.client = self.embedding_pool.clients[0]
        
        # New: Metrics statistics
        self.metrics = metrics
        
        # Init caches
        self.relation_cache = QueryRelationCache(
            cache_file="query_relation_cache.pkl",
            auto_unload_seconds=300,
            metrics=self.metrics
        )
        
        logger.info("KGQueryService initialized successfully with embedding pool")
        
        self.distance_cache = DistanceCache(
            cache_file="distance_cache.pkl",
            auto_unload_seconds=300,
            metrics=self.metrics
        )
        
        logger.info("KGQueryService initialized successfully with embedding pool and distance cache")
    
    def sparql_query(self, sparql_txt):
        try:
            sparql = SPARQLWrapper(self.sparql_endpoint)
            sparql.setReturnFormat(JSON)
            sparql.setQuery(sparql_txt)
            results = sparql.query().convert()
            return results
        except Exception as e:
            # logger.error(f"SPARQL query error: {e}")
            return {}
    
    def process_item(self, item):
        # Check if item is a URL
        if isinstance(item, str) and urlparse(item).scheme in ["http", "https"]:
            return item.split('/')[-1]  # Extract last part
        return item  # If not URL, return original item

    def extract_sparql_variable(self, sparql_str):
        pattern = r"SELECT\s+DISTINCT\s*\?\s*([a-zA-Z][\w]*)"
        match = re.search(pattern, sparql_str, re.IGNORECASE)
        return match.group(1) if match else None

    def search_replaced_entity(self, mid_list):
        replace_mid = []
        get_mid_entity_sparql_format = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?entity
FROM <http://freebase.org/sub_kg>
WHERE {{
ns:{mid} ns:type.object.name ?entity
}}
        """
        for mid in mid_list:
            query_result = self.get_formatted_sparql_data(get_mid_entity_sparql_format.format(mid=mid))
            if 'entity' in query_result:
                replace_mid.append(query_result['entity'][0])
            else:
                replace_mid.append(mid)
        return replace_mid

    def get_formatted_sparql_data(self, sparql_text):
        sparql_results = self.sparql_query(sparql_text)
        results = {}
        if sparql_results:
            for item in sparql_results['results']['bindings']:
                for k, v in item.items():
                    if k not in results:
                        results[k] = []
                    results[k].append(v['value'])
        return results

    def query_full_sparql(self, sparql):
        path_retrieval_results = self.get_formatted_sparql_data(sparql)
        if path_retrieval_results:
            # print(self.extract_sparql_variable(sparql))
            path_retrieval_results = [self.process_item(item) for item in path_retrieval_results[self.extract_sparql_variable(sparql)]]
            path_retrieval_results = self.search_replaced_entity(path_retrieval_results)
        else:
            path_retrieval_results = []
        return path_retrieval_results

    def filter_relation(self, relation):
        """Check if relation is valid: does not start with 'type' or 'common'"""    
        return not (relation.startswith("type") or relation.startswith("freebase.") or relation.startswith("user") or "sameAs" in relation)

    def get_entity_relations_with_sparql(self, node, sparql):
        """Query relations related to node using SPARQL, supporting multiple path patterns"""
        
        # Define actual relation computation function
        def compute_relations():
            # Define SPARQL templates for 4 relation types
            PREFIX = ("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/>\n" )
            KG_LIMITED = "FROM <http://freebase.org/sub_kg>\n"
            get_relation1_format = """{PREFIX}\nSELECT DISTINCT ?relation\n{kg_limited}WHERE {{\n{bgps} {mid} ?relation ?a .\nFILTER EXISTS {{ ?a :type.object.name ?name1 . }}\n}}"""
            
            get_relation2_format = """{PREFIX}\nSELECT DISTINCT ?relation\n{kg_limited}WHERE {{\n{bgps} ?b ?relation {mid} .\nFILTER EXISTS {{ ?b :type.object.name ?name2 . }}\n}}"""
            
            get_relation3_format = """{PREFIX}\nSELECT DISTINCT ?relation\n{kg_limited}WHERE {{\n{bgps}\n{mid} ?relation ?value .\nFILTER (\nregex(str(?value), "^-?\\\\d+$") ||\nregex(str(?value), "^-?\\\\d*\\\\.\\\\d+$") ||\ndatatype(?value) IN (xsd:date, xsd:dateTime, xsd:gYear, xsd:gYearMonth)\n)\n}}"""
            
            get_relation4_format = """{PREFIX}\nSELECT DISTINCT ?relation1 ?relation2\n{kg_limited}WHERE {{\n{bgps} {mid} ?relation1 ?a1 .\n?a1 ?relation2 ?a2 .\nFILTER (?relation1 != :type.object.type) .\nFILTER (?relation2 != :type.object.type) .\nFILTER (?relation1 != ?relation2) .\nFILTER NOT EXISTS {{ ?a1 :type.object.name ?name1 . }}\n}}"""
            
            get_relation5_format = """{PREFIX}\nSELECT DISTINCT ?relation1 ?relation2\n{kg_limited}WHERE {{\n{bgps} ?b1 ?relation1 {mid} .\n?b2 ?relation2 ?b1 .\nFILTER (?relation1 != ?relation2) .\nFILTER NOT EXISTS {{ ?b1 :type.object.name ?name2 . }}\n}}"""
    
            get_relation6_format = '''{PREFIX}\nSELECT DISTINCT ?predicate1 ?predicate2\n{kg_limited}WHERE {{\n{bgps}\n{mid} ?predicate1 ?mid . \n?mid ?predicate2 ?value . \nFILTER (\nregex(str(?value), "^-?\\\\d+$") ||\nregex(str(?value), "^-?\\\\d*\\\\.\\\\d+$") ||\ndatatype(?value) IN (xsd:date, xsd:dateTime, xsd:gYear, xsd:gYearMonth)\n)\nFILTER (?predicate1 != :type.object.type) .\nFILTER (?predicate2 != :type.object.type) .\nFILTER NOT EXISTS {{ ?mid :type.object.name ?name1 . }}\n}}'''
            
            relation_path_list = []
            relation_format = [get_relation1_format, get_relation2_format, get_relation3_format, 
                            get_relation4_format, get_relation5_format, get_relation6_format]
            direction_list = [[1], [-1], [1], [1, 1], [-1, -1], [1, 1]]
            
            # Extract BGP part from SPARQL
            bgps = sparql.strip()
            
            # Execute all types of queries
            relation_results_list = []
            for idx, sparql_format in enumerate(relation_format):
                query = sparql_format.format(PREFIX=PREFIX, kg_limited=KG_LIMITED, bgps=bgps, mid=node)
                # print(query)
                results = self.sparql_query(query)
                relation_results = self.extract_relations_from_query_results(results)
                if relation_results:
                    # print(f"Found {len(relation_results)} relations for pattern {idx+1}")
                    # print(query)
                    pass
                relation_results_list.append(relation_results)
            
            # Build paths for each relation group
            for relation_results, direction in zip(relation_results_list, direction_list):
                paths = self.construct_paths(relation_results, direction, node)
                relation_path_list.append(paths)
            flat_relations = [item for sublist in relation_path_list for item in sublist]
            return list(set(flat_relations))
        
        # Use cache or compute new result
        return self.relation_cache.get_or_compute(
            node=node,
            previous=sparql,
            compute_func=compute_relations,
            forced_use_cache=True
        )

    def extract_relations_from_query_results(self, results):
        """Extract relations from query results"""
        relations_list = []
        if results and 'results' in results and 'bindings' in results['results']:
            for binding in results['results']['bindings']:
                item = []
                for key in sorted(binding.keys()):
                    if 'value' in binding[key]:
                        relation = binding[key]['value']
                        if "http://rdf.freebase.com/ns/" in relation:
                            relation = relation.replace("http://rdf.freebase.com/ns/", "")
                            if "www" not in relation and "type." not in relation and "kg." not in relation and "relation." not in relation and self.filter_relation(relation):
                                item.append(relation)
                        else:
                            break
                if item:
                    relations_list.append(item)
        return relations_list

    def construct_paths(self, relations_groups, dirs, center_entity):
        """Build relation paths"""
        all_paths = []
        letters = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
        # print(dirs)
        for rels in relations_groups:
            # print(rels)
            path = ""
            if rels:
                # current_node = center_entity
                current_node = "<node>"
                for i, (rel, direction) in enumerate(zip(rels, dirs)):
                    # print(f"Processing relation: {rel} with direction {direction}")
                    # Use letters instead of numbers
                    if i < len(letters):
                        next_node = f"?{letters[i]}"
                    else:
                        # After 26 letters, use double letters like ?aa, ?ab ...
                        quotient, remainder = divmod(i, len(letters))
                        next_node = f"?{letters[quotient-1]}{letters[remainder]}"
                    
                    if direction == 1:
                        path += f"{current_node} {rel} {next_node} . "
                    elif direction == -1:
                        path += f"{next_node} {rel} {current_node} . "
                    current_node = next_node
            if dirs == [-1, -1]:
                # print(f"befor reverse {path}")
                path = " . ".join([item.strip() for item in path.strip().split(" .") if item][::-1]) + " ."
                # print(f"after reverse {path}")
            all_paths.append(path.strip())
        return all_paths

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def get_embeddings(self, text, task_description=""):
        """
        Get text embeddings - using parallel connection pool
        
        Note: Strictly ensures return order matches input text list order
        """
        if task_description:
            retrieval_input_text = [self.get_detailed_instruct(
                task_description=task_description,
                query=item
            ) for item in text]
        else:
            retrieval_input_text = text
        if len(text) < 40:
            return self.embedding_pool._get_embeddings_single(
                self.client, retrieval_input_text, model=EMBEDDING_MODEL
            )
        return self.embedding_pool.batch_embed_parallel(
            texts=retrieval_input_text,
            model=EMBEDDING_MODEL
        )

    def get_topk_arr_index(self, arr, k=30, threshold=0.0):
        sorted_indices = np.argsort(arr)[::-1]
        filtered_indices = [i for i in sorted_indices if arr[i] >= threshold]
        return filtered_indices[:k]

    def get_topk_relation(self, question, relation_list, k=30, threshold=0.0):
        question_task_description = "Given a question and the relationships in the knowledge graph, select the relationships that are most relevant to the current question."
        # question_task_description = "Given a reasoning statement that explains what information needs to be found next in order to answer a question, identify which relation paths in a knowledge graph are most useful for retrieving or inferring that information. The goal is to match reasoning intent with the most semantically relevant relation path."
        # print(f"Filtering {len(relation_list)} relations for question: {question}")
        q_embedding = self.get_embeddings([question], task_description=question_task_description).squeeze(0)
        rp_embeddings = self.get_embeddings(relation_list, task_description="")
        similarity_scores = q_embedding @ rp_embeddings.T
        top_indices = self.get_topk_arr_index(similarity_scores, k=k, threshold=threshold)
        return [relation_list[i] for i in top_indices]

    # def retrieval_related_relation_list(self, question, relation_list, k=20, threshold=0.0):
    #     if len(relation_list) >= k:
    #         related_relation_list = self.get_topk_relation(question, relation_list, k=k, threshold=threshold)
    #     else:
    #         related_relation_list = relation_list
    #     return related_relation_list

    def retrieval_related_relation_list(self, question, relation_list, k=20, threshold=0.0):
        return self.get_topk_relation(question, relation_list, k=k, threshold=threshold) if relation_list else []

    def extract_question_from_context(self, context):
        """Extract question from context"""
        # Assume format "Question: ..."
        question_match = re.search(r"Question:\s*(.+?)(?:\n|$)", context)
        if question_match:
            return question_match.group(1).strip()
        return ""

    def insert_from_clause(self, sparql_query: str, graph_iri: str) -> str:
        lines = sparql_query.splitlines()
        new_lines = []
        from_inserted = False

        for line in lines:
            new_lines.append(line)
            if not from_inserted and line.strip().upper().startswith("SELECT"):
                new_lines.append(f"FROM <{graph_iri}>")
                from_inserted = True

        return "\n".join(new_lines)
    
    def limit_sparql_results(self, sparql_query: str, limit: int = 10) -> str:
        if re.search(r'\bLIMIT\b', sparql_query, re.IGNORECASE):
            return sparql_query  # Already has LIMIT clause, return directly
        else:
            return sparql_query.strip() + f"\nLIMIT {limit}"

    def extract_entities(self, sparql):
        return re.findall(r'"([^"]+)"', sparql)

    def get_id_for_entity(self, entity):
        query = f'''
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?id
        WHERE {{
            ?id ns:type.object.name "{entity}" .
        }}
        '''
        result = get_formatted_sparql_data(query)  # Your function
        if result and "id" in result:
            return result["id"][0]
        return None

    def replace_entities_with_ids(self, original_sparql):
        entities = self.extract_entities(original_sparql)
        updated_sparql = original_sparql

        for ent in entities:
            ent_id = self.get_id_for_entity(ent)
            if ent_id:
                # Replace only with quotes
                updated_sparql = updated_sparql.replace(f'"{ent}"', f":{ent_id.replace("http://rdf.freebase.com/ns/", "")}")
            else:
                # No ID found → keep original, no replace
                pass

        return updated_sparql

    def fix_date_filter(self, sparql_query):
        pattern = re.compile(
            r'FILTER\s*\(\s*(\?\w+)\s*(>|<|>=|<=|=)\s*"([^"]+)"\^\^xsd:dateTime\s*\)',
            re.IGNORECASE
        )
        def repl(match):
            var = match.group(1)        # ?Variable
            op = match.group(2)         # Operator > < >= <= =
            date_value = match.group(3) # Date string
            return f'FILTER ( xsd:date({var}) {op} "{date_value}"^^xsd:dateTime )'
        return pattern.sub(repl, sparql_query)

    def process_query(self, query_data):
        """Process single query"""
        start_time = time.time()
        q_type = query_data["type"]
        q_content = query_data["content"]
        params = query_data["parameters"]
        idx = params["idx"]
        # entity_query = query_data["entity_query"]
        entity_query = query_data.get("entity_query", True)
        # logger.info(f"Processing query - Type: {q_type}, IDX: {idx}")
        
        try:
            if q_type == "sparql":
                if self.metrics:
                    self.metrics.record_query_execution('sparql')
                # if ":m." in q_content or ":g." in q_content:
                q_content = self.insert_from_clause(q_content, "http://freebase.org/sub_kg")
                q_content = self.limit_sparql_results(q_content, limit=100)
                if entity_query:
                    q_content = self.replace_entities_with_ids(q_content)
                q_content = self.fix_date_filter(q_content)
                results = self.query_full_sparql(q_content)
                results = list(set(results))[:10]
                results = [item.replace("http://rdf.freebase.com/ns/", "") for item in results]
                results = [item for item in results if not item.startswith("m.") or not item.startswith("g.")]
                response_time = time.time() - start_time
                if self.metrics:
                    self.metrics.record_request(q_type, success=True, response_time=response_time)
                
                # logger.info(f"SPARQL query completed - IDX: {idx}, Results: {len(results)}, Time: {response_time:.3f}s")
                
                return {
                    "type": "sparql", 
                    "results": results,
                    "query": q_content,
                    "idx": idx
                }
                # else:
                #     response_time = time.time() - start_time
                #     if self.metrics:
                #         self.metrics.record_request(q_type, success=True, response_time=response_time)
                #     return {
                #         "type": "sparql", 
                #         "results": [],
                #         "query": q_content,
                #         "idx": idx
                #     }
            elif q_type == "node":
                if self.metrics:
                    self.metrics.record_query_execution('node')
                
                node = q_content
                previous_sparql = params.get("previous_sparql", "")
                question = params.get("question", "")
                filter_k = params.get("filter_k", 40)
                filter_threshold = params.get("filter_threshold", 0.0)
                # logger.info(f"Node query started - Node: {node}, IDX: {idx}")
                # Choose query method based on node type
                if node.startswith("m.") or node.startswith("g."):
                    if node.startswith(":"):
                        relations = self.get_entity_relations_with_sparql(node=node, sparql="")
                    else:
                        relations = self.get_entity_relations_with_sparql(node=":" + node, sparql="")
                elif node.startswith("?"):
                    if previous_sparql and node in previous_sparql:
                        if entity_query:
                            previous_sparql = self.replace_entities_with_ids(previous_sparql)
                        relations = self.get_entity_relations_with_sparql(node=node, sparql=previous_sparql)
                    else:
                        relations = []
                else:
                    # Query relations based on entity name
                    if isinstance(node, str):
                        entity_id = self.get_id_for_entity(node)
                        # logger.info(f"Resolved entity '{node}' to ID: {entity_id}")
                        if entity_id:
                            relations = self.get_entity_relations_with_sparql(node=":" + entity_id.replace("http://rdf.freebase.com/ns/", ""), sparql="")
                        else:
                            relations = []
                    else:
                        relations = []
                # print(f"Retrieved {len(relations)} relation paths for node {node}")
                # Process results - add relation filtering
                if relations:
                    flat_relations = [r for r in relations if r]
                    filter_applied = False
                    
                    if question and flat_relations:
                        filtered_relations = self.retrieval_related_relation_list(
                            question, flat_relations, k=filter_k, threshold=filter_threshold
                        )
                        filter_applied = True
                    else:
                        filtered_relations = flat_relations
                    
                    # Record relation retrieval stats
                    if self.metrics:
                        self.metrics.record_relation_retrieval(
                            len(flat_relations), 
                            len(filtered_relations),
                            filter_applied
                        )
                    
                    response_time = time.time() - start_time
                    if self.metrics:
                        self.metrics.record_request(q_type, success=True, response_time=response_time)
                    
                    # logger.info(f"Node query completed - Node: {node}, IDX: {idx}, "
                    #           f"Original: {len(flat_relations)}, Filtered: {len(filtered_relations)}, "
                    #           f"Time: {response_time:.3f}s")
                    
                    return {
                        "type": "node", 
                        "node": node,
                        "results": filtered_relations,
                        "original_count": len(flat_relations),
                        "filtered_count": len(filtered_relations),
                        "content": question,
                        "idx": idx
                    }
                else:
                    response_time = time.time() - start_time
                    if self.metrics:
                        self.metrics.record_request(q_type, success=True, response_time=response_time)
                    
                    # logger.info(f"Node query completed - Node: {node}, IDX: {idx}, No relations found")
                    
                    return {
                        "type": "node", 
                        "node": node,
                        "results": [],
                        "original_count": 0,
                        "filtered_count": 0,
                        "content": question,
                        "idx": idx
                    }
            else:
                response_time = time.time() - start_time
                if self.metrics:
                    self.metrics.record_request(q_type, success=False, response_time=response_time, 
                                              error_msg="Unknown query type")
                
                # logger.warning(f"Unknown query type: {q_type}, IDX: {idx}")
                
                return {
                    "type": "error", 
                    "message": "Unknown query type",
                    "idx": idx
                }
        except Exception as e:
            response_time = time.time() - start_time
            if self.metrics:
                self.metrics.record_request(q_type, success=False, response_time=response_time, 
                                          error_msg=str(e))
            
            logger.error(f"Error processing query - Type: {q_type}, IDX: {idx}, Error: {str(e)}")
            
            return {
                "type": "error", 
                "message": str(e),
                "idx": idx
            }
    # ============ New: Distance Query Methods ============
    def get_distance_between_entities(self, entity_a, entity_b, max_distance=4, return_cache_flag=False):
        """
        Query distance between two entities (with cache)
        When return_cache_flag=True, return (distance, from_cache)
        """
        cached_distance = self.distance_cache.get(entity_a, entity_b)
        if cached_distance is not None:
            if return_cache_flag:
                return cached_distance, True
            return cached_distance

        if max_distance > 3:
            query_distance_sparql_format = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?dist
WHERE {{
  ?start ns:type.object.name "{}" .
  ?end ns:type.object.name "{}" .
  {{
    ?start (ns:compressed_edge|^ns:compressed_edge) ?end .
    BIND(1 AS ?dist)
  }}
  UNION
  {{
    ?start (ns:compressed_edge|^ns:compressed_edge)/
            (ns:compressed_edge|^ns:compressed_edge)
            ?end .
    BIND(2 AS ?dist)
  }}
  UNION
  {{
    ?start (ns:compressed_edge|^ns:compressed_edge)/
            (ns:compressed_edge|^ns:compressed_edge)/
            (ns:compressed_edge|^ns:compressed_edge)
            ?end .
    BIND(3 AS ?dist)
  }}
}}
ORDER BY ?dist
LIMIT 1
"""
        else:
            query_distance_sparql_format = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?dist
WHERE {{
  ?start ns:type.object.name "{}" .
  ?end ns:type.object.name "{}" .
  {{
    ?start (ns:compressed_edge|^ns:compressed_edge) ?end .
    BIND(1 AS ?dist)
  }}
  UNION
  {{
    ?start (ns:compressed_edge|^ns:compressed_edge)/
            (ns:compressed_edge|^ns:compressed_edge)
            ?end .
    BIND(2 AS ?dist)
  }}
}}
ORDER BY ?dist
LIMIT 1
"""

        query = query_distance_sparql_format.format(entity_a, entity_b)
        try:
            res = self.get_formatted_sparql_data(query)
        except Exception as e:
            logger.error(f"Distance query error for ({entity_a}, {entity_b}): {e}")
            res = {}

        if res and 'dist' in res and res['dist']:
            try:
                distance = int(res['dist'][0])
            except Exception:
                distance = max_distance
        else:
            distance = max_distance

        if distance < 1:
            distance = 1
        if distance > max_distance:
            distance = max_distance

        self.distance_cache.set(entity_a, entity_b, distance)

        if return_cache_flag:
            return distance, False
        return distance
    
    def get_distance_between_sets_voting(self, set_a, set_b, max_distance_if_no_result=4, early_stop_global_min=1):
        """
        Query distance between two sets of entities using voting mechanism
        """
        per_a = {}
        overall_min = max_distance_if_no_result
        cache_hits = 0
        actual_queries = 0
        early_stopped = False

        half_threshold = (len(set_b) // 2) + 1 if set_b else 0

        for a in set_a:
            votes = defaultdict(int)

            if not set_b:
                per_a[a] = {'distance': max_distance_if_no_result, 'votes': {}}
                overall_min = min(overall_min, max_distance_if_no_result)
                continue

            for b in set_b:
                dist, from_cache = self.get_distance_between_entities(a, b, max_distance_if_no_result, return_cache_flag=True)
                votes[dist] += 1
                if from_cache:
                    cache_hits += 1
                else:
                    actual_queries += 1

                if votes[dist] >= half_threshold:
                    a_distance = dist
                    break
            else:
                # Not break
                best_d = None
                best_cnt = -1
                for d_val, cnt in votes.items():
                    if cnt > best_cnt or (cnt == best_cnt and (best_d is None or d_val < best_d)):
                        best_cnt = cnt
                        best_d = d_val
                a_distance = best_d if best_d is not None else max_distance_if_no_result

            per_a[a] = {'distance': a_distance, 'votes': dict(votes)}
            overall_min = min(overall_min, a_distance)

            if early_stop_global_min is not None and overall_min <= early_stop_global_min:
                early_stopped = True
                break

        if self.metrics:
            self.metrics.record_distance_query(
                set_a_size=len(set_a),
                set_b_size=len(set_b),
                overall_distance=overall_min,
                cache_hits=cache_hits,
                actual_queries=actual_queries,
                early_stopped=early_stopped
            )

        return {'per_a': per_a, 'distance': overall_min}


@app.route('/kg_query', methods=['POST'])
def handle_kg_query():
    """Receive single KG query request"""
    try:
        data = request.json
        query = data.get('query', {})
        
        # logger.info(f"Received single query - Type: {query.get('type', 'unknown')}")
        
        result = kg_service.process_query(query)
        return jsonify({
            'status': 'success',
            'results': result
        })
    except Exception as e:
        logger.error(f"Single query error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============ New: Monitoring Endpoints ============
@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get service runtime metrics"""
    stats = metrics.get_stats()
    return jsonify(stats)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    stats = metrics.get_stats()
    return jsonify({
        'status': 'healthy',
        'uptime': stats['uptime_formatted'],
        'total_requests': stats['total_requests']
    })

@app.route('/entity_distance', methods=['POST'])
def handle_entity_distance():
    """Query distance between entity sets"""
    start_time = time.time()
    
    try:
        data = request.json
        set_a = data.get('set_a', [])
        set_b = data.get('set_b', [])
        max_distance = data.get('max_distance', 4)
        early_stop = data.get('early_stop_global_min', 1)
        
        if not set_a:
            return jsonify({
                'status': 'error',
                'message': 'set_a cannot be empty'
            }), 400
        
        # logger.info(f"Processing distance query - Set A: {len(set_a)} entities, Set B: {len(set_b)} entities")
        
        result = kg_service.get_distance_between_sets_voting(
            set_a=set_a,
            set_b=set_b,
            max_distance_if_no_result=max_distance,
            early_stop_global_min=early_stop
        )
        
        response_time = time.time() - start_time
        
        if metrics:
            metrics.record_request('distance', success=True, response_time=response_time)
        
        # logger.info(f"Distance query completed - Overall distance: {result['distance']}, Time: {response_time:.3f}s")
        
        return jsonify({
            'status': 'success',
            'distance': result['distance'],
            'details': result['per_a'],
            'response_time': f"{response_time:.3f}s"
        })
        
    except Exception as e:
        response_time = time.time() - start_time
        
        if metrics:
            metrics.record_request('distance', success=False, response_time=response_time, error_msg=str(e))
        
        # logger.error(f"Distance query error: {str(e)}")
        
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

"""
# 查看实时日志
tail -f kg_query_service.log

# 定期检查指标
curl http://localhost:5501/metrics

# 健康检查
curl http://localhost:5501/health
"""


if __name__ == '__main__':
    # Init metrics
    metrics = ServiceMetrics()
    kg_service = KGQueryService(metrics=metrics)
    
    server_port = int(os.getenv("SERVER_PORT", 5501))
    logger.info(f"Starting KG Query Server on port {server_port}")

    # Disable Werkzeug logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    logging.disable(logging.INFO)

    app.run(host='0.0.0.0', port=server_port, debug=False)