import numpy as np
import faiss
from typing import Tuple, List
import time

class IndexEvaluator:
    def __init__(self, index: faiss.Index, ground_truth_data: np.ndarray):
        self.index = index
        self.ground_truth = ground_truth_data
        
    def evaluate_recall(self, queries: np.ndarray, k: int, n_ground_truth: int) -> float:
        """
        Evaluate recall@k metric.
        
        Args:
            queries: Query vectors to evaluate
            k: Number of nearest neighbors to retrieve
            n_ground_truth: Number of ground truth neighbors to compare against
            
        Returns:
            Average recall score
        """
        # Get ground truth neighbors using exact search
        exact_index = faiss.IndexFlatL2(self.ground_truth.shape[1])
        exact_index.add(self.ground_truth)
        ground_truth_distances, ground_truth_ids = exact_index.search(queries, n_ground_truth)
        
        # Get approximate neighbors from FAISS index
        approx_distances, approx_ids = self.index.search(queries, k)
        
        # Calculate recall
        recall_scores = []
        for i in range(len(queries)):
            gt_set = set(ground_truth_ids[i])
            pred_set = set(approx_ids[i])
            recall = len(gt_set.intersection(pred_set)) / len(gt_set)
            recall_scores.append(recall)
            
        return np.mean(recall_scores)
    
    def evaluate_latency(self, queries: np.ndarray, k: int, n_runs: int = 100) -> Tuple[float, float]:
        """
        Evaluate search latency.
        
        Returns:
            Tuple of (average_latency, std_deviation)
        """
        latencies = []
        for _ in range(n_runs):
            start_time = time.time()
            self.index.search(queries, k)
            latencies.append(time.time() - start_time)
            
        return np.mean(latencies), np.std(latencies)
    
    def evaluate_memory(self) -> int:
        """
        Evaluate memory usage of the index in bytes.
        """
        # For IndexFlatL2, each vector uses dimension * 4 bytes (float32)
        # return self.index.ntotal * self.index.d * 4
        return self.index.get_memory_usage()
    
    def evaluate_precision(self, queries: np.ndarray, k: int, n_ground_truth: int) -> float:
        """
        Evaluate precision@k metric.
        """
        exact_index = faiss.IndexFlatL2(self.ground_truth.shape[1])
        exact_index.add(self.ground_truth)
        ground_truth_distances, ground_truth_ids = exact_index.search(queries, n_ground_truth)
        
        approx_distances, approx_ids = self.index.search(queries, k)
        
        precision_scores = []
        for i in range(len(queries)):
            gt_set = set(ground_truth_ids[i])
            pred_set = set(approx_ids[i])
            precision = len(gt_set.intersection(pred_set)) / k
            precision_scores.append(precision)
            
        return np.mean(precision_scores)

    def get_index_stats(self) -> dict:
        """
        Get basic statistics about the index.
        """
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "is_trained": self.index.is_trained,
            "index_type": type(self.index).__name__
        }
        return stats

# Example usage:
def evaluate_faiss_index(index_path: str, feature_matrix: np.ndarray, test_queries: np.ndarray):
    """
    Comprehensive evaluation of a FAISS index
    """
    # Load index
    index = faiss.read_index(index_path)
    
    # Initialize evaluator
    evaluator = IndexEvaluator(index, feature_matrix)
    
    # Run evaluations
    k = 10  # number of neighbors to retrieve
    n_ground_truth = 10  # number of ground truth neighbors
    
    results = {
        "stats": evaluator.get_index_stats(),
        "recall": evaluator.evaluate_recall(test_queries, k, n_ground_truth),
        "precision": evaluator.evaluate_precision(test_queries, k, n_ground_truth),
        "latency": evaluator.evaluate_latency(test_queries, k),
        "memory_usage": evaluator.evaluate_memory()
    }
    
    return results