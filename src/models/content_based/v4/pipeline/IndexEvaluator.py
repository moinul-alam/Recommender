import faiss
import numpy as np
import time

class IndexEvaluator:
    def __init__(self, index: faiss.Index, ground_truth: np.ndarray):
        """
        Initialize the evaluator with the FAISS index and the original dataset (ground truth).
        """
        self.index = index
        self.ground_truth = ground_truth

    def evaluate_recall(self, queries: np.ndarray, k: int, n_ground_truth: int) -> float:
        """
        Compute Recall@k.
        """
        exact_index = faiss.IndexFlatL2(self.ground_truth.shape[1])
        exact_index.add(self.ground_truth)
        _, ground_truth_ids = exact_index.search(queries, n_ground_truth)

        _, approx_ids = self.index.search(queries, k)

        recall = np.mean([
            len(set(approx_ids[i]) & set(ground_truth_ids[i])) / len(set(ground_truth_ids[i]))
            for i in range(len(queries))
        ])
        return recall

    def evaluate_precision(self, queries: np.ndarray, k: int, n_ground_truth: int) -> float:
        """
        Compute Precision@k.
        """
        exact_index = faiss.IndexFlatL2(self.ground_truth.shape[1])
        exact_index.add(self.ground_truth)
        _, ground_truth_ids = exact_index.search(queries, n_ground_truth)

        _, approx_ids = self.index.search(queries, k)

        precision = np.mean([
            len(set(approx_ids[i]) & set(ground_truth_ids[i])) / k
            for i in range(len(queries))
        ])
        return precision

    def evaluate_map(self, queries: np.ndarray, k: int, n_ground_truth: int) -> float:
        """
        Compute Mean Average Precision (mAP@k).
        """
        exact_index = faiss.IndexFlatL2(self.ground_truth.shape[1])
        exact_index.add(self.ground_truth)
        _, ground_truth_ids = exact_index.search(queries, n_ground_truth)

        _, approx_ids = self.index.search(queries, k)

        ap_scores = []
        for i in range(len(queries)):
            gt_set = set(ground_truth_ids[i])
            retrieved = approx_ids[i]

            num_hits = 0
            avg_precision = 0.0

            for j, item in enumerate(retrieved):
                if item in gt_set:
                    num_hits += 1
                    avg_precision += num_hits / (j + 1)

            ap_scores.append(avg_precision / len(gt_set) if len(gt_set) > 0 else 0)

        return np.mean(ap_scores)

    def evaluate_ndcg(self, queries: np.ndarray, k: int, n_ground_truth: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG@k).
        """
        exact_index = faiss.IndexFlatL2(self.ground_truth.shape[1])
        exact_index.add(self.ground_truth)
        _, ground_truth_ids = exact_index.search(queries, n_ground_truth)

        _, approx_ids = self.index.search(queries, k)

        ndcg_scores = []
        for i in range(len(queries)):
            gt_list = list(ground_truth_ids[i])
            retrieved = approx_ids[i]

            dcg = 0.0
            for j, item in enumerate(retrieved):
                if item in gt_list:
                    rel = 1 / (gt_list.index(item) + 1)  # Relevance score
                    dcg += rel / np.log2(j + 2)

            idcg = sum(1 / np.log2(j + 2) for j in range(min(len(gt_list), k)))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0)

        return np.mean(ndcg_scores)

    def evaluate_query_coverage(self, queries: np.ndarray, k: int) -> float:
        """
        Compute the percentage of queries that return at least one valid result.
        """
        _, approx_ids = self.index.search(queries, k)
        covered_queries = sum(1 for results in approx_ids if len(set(results)) > 0)
        return covered_queries / len(queries)

    def evaluate_compression_ratio(self, original_dim: int) -> float:
        """
        Compute the compression ratio of the FAISS index.
        """
        original_size = self.ground_truth.shape[0] * original_dim * 4  # float32 size
        compressed_size = self.evaluate_memory()  # Use the updated memory calculation
        return original_size / compressed_size if compressed_size > 0 else 0

    def evaluate_latency(self, queries: np.ndarray, k: int, num_trials: int = 10) -> float:
        """
        Measure the average search latency per query (in milliseconds).
        """
        start_time = time.time()
        for _ in range(num_trials):
            self.index.search(queries, k)
        end_time = time.time()
        return (end_time - start_time) / (num_trials * len(queries)) * 1000  # Convert to ms

    def evaluate_memory(self) -> int:
        """
        Evaluate memory usage of the index in bytes.
        """
        memory_usage = self.index.ntotal * self.index.d * 4  # 4 bytes per float32
        return memory_usage

    def get_index_stats(self)-> dict:
        """
        Get basic index statistics (size, type, and memory usage).
        """
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "is_trained": self.index.is_trained,
            "index_type": type(self.index).__name__
        }
        return stats
