# -*- coding: utf-8 -*-
"""
Order Book Vectorizer - Live CPU-Bound and GPU-Accelerated Module.

Converts order book snapshots to normalized bitwise vectors for
trading signal generation and strategy bitmapping.

Supports 16-bit/32-bit vectorization, CPU and GPU batch processing.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from core.unified_math_system import unified_math

logger = logging.getLogger(__name__)

# Try to import cupy for GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("CuPy GPU acceleration is available for batch vectorization.")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    logger.info("CuPy not found. GPU batch vectorization will use CPU fallback.")


class OrderBookVectorizer:
    """
    Order Book Vectorizer for real-time trading signal generation.
    
    Converts order book snapshots to normalized bitwise vectors
    with support for 16-bit and 32-bit vectorization.
    Supports CPU and GPU batch processing.
    """
    
    def __init__(
        self,
        default_bit_depth: int = 16,
        enable_gpu_acceleration: bool = False,
        normalization_method: str = "minmax",
        quantization_bits: int = 8,
    ):
        """Initialize the order book vectorizer.
        
        Args:
            default_bit_depth: Default vector size (16 or 32)
            enable_gpu_acceleration: Enable GPU acceleration for batched processing
            normalization_method: Normalization method ('minmax', 'zscore', 'robust')
            quantization_bits: Bits per vector element (8-bit default)
        """
        self.default_bit_depth = default_bit_depth
        self.enable_gpu_acceleration = enable_gpu_acceleration and GPU_AVAILABLE
        self.normalization_method = normalization_method
        self.quantization_bits = quantization_bits
        
        # Validation
        if default_bit_depth not in [16, 32]:
            raise ValueError("bit_depth must be 16 or 32")
        if quantization_bits not in [8, 16]:
            raise ValueError("quantization_bits must be 8 or 16")
        
        # Performance tracking
        self.vectorization_stats = {
            "total_vectorizations": 0,
            "avg_processing_time": 0.0,
            "successful_vectorizations": 0,
            "failed_vectorizations": 0,
            "gpu_enabled": self.enable_gpu_acceleration,
        }
        
        logger.info(
            f"OrderBookVectorizer initialized: "
            f"bit_depth={default_bit_depth}, "
            f"gpu_acceleration={self.enable_gpu_acceleration}, "
            f"normalization={normalization_method}, "
            f"quantization={quantization_bits}-bit"
        )
    
    def vectorize_order_book(
        self, 
        order_book: Dict[str, List], 
        bit_depth: Optional[int] = None,
        symbol: str = "BTC/USDC"
    ) -> np.ndarray:
        """
        Convert order book snapshot to a normalized bitwise vector.
        
        Args:
            order_book: dict with 'bids' and 'asks' (each is [price, volume])
            bit_depth: Total vector size (must be even, half for bids, half for asks)
            symbol: Trading symbol for logging
            
        Returns:
            np.ndarray: Quantized vectorized snapshot of order book
        """
        start_time = time.time()
        bit_depth = bit_depth or self.default_bit_depth
        
        try:
            # Validate bit depth
            assert bit_depth % 2 == 0, f"bit_depth must be even, got {bit_depth}"
            
            half_depth = bit_depth // 2
            bids = order_book.get("bids", [])[:half_depth]
            asks = order_book.get("asks", [])[:half_depth]
            
            # Extract prices (ignore volumes for now)
            bid_prices = [float(b[0]) for b in bids]
            ask_prices = [float(a[0]) for a in asks]
            
            # Pad with zeros if insufficient data
            while len(bid_prices) < half_depth:
                bid_prices.append(bid_prices[-1] if bid_prices else 0.0)
            while len(ask_prices) < half_depth:
                ask_prices.append(ask_prices[-1] if ask_prices else 0.0)
            
            # Combine into single vector
            vector = np.array(bid_prices + ask_prices, dtype=np.float64)
            
            # Normalize vector
            normalized_vector = self._normalize_vector(vector)
            
            # Quantize to specified bit depth
            quantized_vector = self._quantize_vector(normalized_vector)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(True, processing_time)
            
            logger.debug(
                f"Order book vectorized for {symbol}: "
                f"bit_depth={bit_depth}, "
                f"processing_time={processing_time:.6f}s"
            )
            
            return quantized_vector
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(False, processing_time)
            logger.error(f"Order book vectorization failed for {symbol}: {e}")
            
            # Return zero vector as fallback
            return np.zeros(bit_depth, dtype=np.uint8)

    def vectorize_order_book_batch(
        self,
        order_books: List[Dict[str, List]],
        symbols: List[str],
        bit_depth: Optional[int] = None,
        use_gpu: Optional[bool] = None
    ) -> Dict[str, np.ndarray]:
        """
        Vectorize multiple order books in batch, optionally using GPU.
        
        Args:
            order_books: List of order book dictionaries
            symbols: List of corresponding symbols
            bit_depth: Bit depth for vectorization
            use_gpu: If True, use GPU (if available); if False, use CPU; if None, use self.enable_gpu_acceleration
        Returns:
            Dict mapping symbols to their vectorized order books
        """
        start_time = time.time()
        results = {}
        bit_depth = bit_depth or self.default_bit_depth
        use_gpu = self.enable_gpu_acceleration if use_gpu is None else use_gpu

        if use_gpu and GPU_AVAILABLE:
            # GPU batch processing
            try:
                # Prepare price arrays for all order books
                price_matrix = []
                for order_book in order_books:
                    half_depth = bit_depth // 2
                    bids = order_book.get("bids", [])[:half_depth]
                    asks = order_book.get("asks", [])[:half_depth]
                    bid_prices = [float(b[0]) for b in bids]
                    ask_prices = [float(a[0]) for a in asks]
                    while len(bid_prices) < half_depth:
                        bid_prices.append(bid_prices[-1] if bid_prices else 0.0)
                    while len(ask_prices) < half_depth:
                        ask_prices.append(ask_prices[-1] if ask_prices else 0.0)
                    price_vector = bid_prices + ask_prices
                    price_matrix.append(price_vector)
                price_matrix = np.array(price_matrix, dtype=np.float64)
                # Move to GPU
                price_matrix_gpu = cp.asarray(price_matrix)
                # Normalize on GPU
                min_vals = cp.min(price_matrix_gpu, axis=1, keepdims=True)
                max_vals = cp.max(price_matrix_gpu, axis=1, keepdims=True)
                ptp_vals = max_vals - min_vals
                ptp_vals = cp.where(ptp_vals < 1e-9, 1.0, ptp_vals)
                normalized_gpu = (price_matrix_gpu - min_vals) / ptp_vals
                # Quantize on GPU
                max_value = (2 ** self.quantization_bits) - 1
                quantized_gpu = (normalized_gpu * max_value).astype(cp.uint8)
                quantized_cpu = cp.asnumpy(quantized_gpu)
                # Map back to symbols
                for i, symbol in enumerate(symbols):
                    results[symbol] = quantized_cpu[i]
                batch_time = time.time() - start_time
                logger.info(f"GPU batch vectorization completed: {len(results)} symbols in {batch_time:.3f}s")
                return results
            except Exception as e:
                logger.error(f"GPU batch vectorization failed, falling back to CPU: {e}")
                use_gpu = False  # Fallback to CPU
        # CPU fallback
        for order_book, symbol in zip(order_books, symbols):
            try:
                vector = self.vectorize_order_book(order_book, bit_depth, symbol)
                results[symbol] = vector
            except Exception as e:
                logger.error(f"Batch vectorization failed for {symbol}: {e}")
                results[symbol] = np.zeros(bit_depth, dtype=np.uint8)
        batch_time = time.time() - start_time
        logger.info(f"CPU batch vectorization completed: {len(results)} symbols in {batch_time:.3f}s")
        return results

    def visualize_batch_comparison(
        self,
        order_books: List[Dict[str, List]],
        symbols: List[str],
        bit_depth: Optional[int] = None
    ) -> None:
        """
        Visualize CPU vs GPU batch vectorization results for the same order books.
        Shows side-by-side heatmaps for each symbol.
        """
        import matplotlib.pyplot as plt
        bit_depth = bit_depth or self.default_bit_depth
        cpu_results = self.vectorize_order_book_batch(order_books, symbols, bit_depth, use_gpu=False)
        gpu_results = self.vectorize_order_book_batch(order_books, symbols, bit_depth, use_gpu=True) if GPU_AVAILABLE else None
        n = len(symbols)
        fig, axes = plt.subplots(n, 2 if gpu_results else 1, figsize=(8 if gpu_results else 4, 2*n))
        if n == 1:
            axes = [axes] if not gpu_results else [axes[0], axes[1]]
        for i, symbol in enumerate(symbols):
            cpu_vec = cpu_results[symbol].reshape(2, bit_depth//2)
            ax = axes[i][0] if gpu_results else axes[i]
            im = ax.imshow(cpu_vec, cmap='viridis')
            ax.set_title(f"CPU: {symbol}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if gpu_results:
                gpu_vec = gpu_results[symbol].reshape(2, bit_depth//2)
                ax2 = axes[i][1]
                im2 = ax2.imshow(gpu_vec, cmap='plasma')
                ax2.set_title(f"GPU: {symbol}")
                plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector using specified method."""
        if self.normalization_method == "minmax":
            return self._minmax_normalize(vector)
        elif self.normalization_method == "zscore":
            return self._zscore_normalize(vector)
        elif self.normalization_method == "robust":
            return self._robust_normalize(vector)
        else:
            return self._minmax_normalize(vector)
    
    def _minmax_normalize(self, vector: np.ndarray) -> np.ndarray:
        """Min-max normalization."""
        if vector.ptp() < 1e-9:
            return np.zeros_like(vector)
        return (vector - vector.min()) / (vector.max() - vector.min())
    
    def _zscore_normalize(self, vector: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        mean = np.mean(vector)
        std = np.std(vector)
        if std < 1e-9:
            return np.zeros_like(vector)
        return (vector - mean) / std
    
    def _robust_normalize(self, vector: np.ndarray) -> np.ndarray:
        """Robust normalization using median and MAD."""
        median = np.median(vector)
        mad = np.median(np.abs(vector - median))
        if mad < 1e-9:
            return np.zeros_like(vector)
        return (vector - median) / mad
    
    def _quantize_vector(self, normalized_vector: np.ndarray) -> np.ndarray:
        """Quantize normalized vector to specified bit depth."""
        max_value = (2 ** self.quantization_bits) - 1
        quantized = (normalized_vector * max_value).astype(np.uint8)
        return quantized
    
    def compute_vector_metrics(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics for a vectorized order book.
        
        Args:
            vector: Vectorized order book
            
        Returns:
            Dictionary of computed metrics
        """
        try:
            half_depth = len(vector) // 2
            bid_vector = vector[:half_depth]
            ask_vector = vector[half_depth:]
            
            metrics = {
                "bid_mean": float(np.mean(bid_vector)),
                "ask_mean": float(np.mean(ask_vector)),
                "bid_std": float(np.std(bid_vector)),
                "ask_std": float(np.std(ask_vector)),
                "spread": float(np.mean(ask_vector) - np.mean(bid_vector)),
                "volatility": float(np.std(vector)),
                "entropy": float(self._compute_entropy(vector)),
                "skewness": float(self._compute_skewness(vector)),
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Vector metrics computation failed: {e}")
            return {}
    
    def _compute_entropy(self, vector: np.ndarray) -> float:
        """Compute Shannon entropy of the vector."""
        try:
            # Normalize to probability distribution
            vector_norm = np.abs(vector).astype(float)
            if np.sum(vector_norm) == 0:
                return 0.0
            vector_norm = vector_norm / np.sum(vector_norm)
            
            # Compute entropy
            entropy = -np.sum(vector_norm * np.log2(vector_norm + 1e-12))
            return entropy
        except Exception:
            return 0.0
    
    def _compute_skewness(self, vector: np.ndarray) -> float:
        """Compute skewness of the vector."""
        try:
            mean = np.mean(vector)
            std = np.std(vector)
            if std == 0:
                return 0.0
            skewness = np.mean(((vector - mean) / std) ** 3)
            return skewness
        except Exception:
            return 0.0
    
    def _update_stats(self, success: bool, processing_time: float) -> None:
        """Update vectorization statistics."""
        self.vectorization_stats["total_vectorizations"] += 1
        
        if success:
            self.vectorization_stats["successful_vectorizations"] += 1
        else:
            self.vectorization_stats["failed_vectorizations"] += 1
        
        # Update average processing time
        total_time = self.vectorization_stats["avg_processing_time"] * (
            self.vectorization_stats["total_vectorizations"] - 1
        )
        self.vectorization_stats["avg_processing_time"] = (
            (total_time + processing_time) / self.vectorization_stats["total_vectorizations"]
        )
    
    def get_performance_summary(self) -> Dict[str, Union[int, float]]:
        """Get vectorization performance summary."""
        return self.vectorization_stats.copy()


def vectorize_order_book(
    order_book: Dict[str, List], 
    bit_depth: int = 16
) -> np.ndarray:
    """
    Standalone function for order book vectorization.
    
    Args:
        order_book: dict with 'bids' and 'asks' (each is [price, volume])
        bit_depth: Total vector size (must be even, half for bids, half for asks)
        
    Returns:
        np.ndarray: Quantized vectorized snapshot of order book
    """
    vectorizer = OrderBookVectorizer(default_bit_depth=bit_depth)
    return vectorizer.vectorize_order_book(order_book, bit_depth)


# Global instance for easy access
order_book_vectorizer = OrderBookVectorizer()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Sample order book data
    sample_order_book = {
        "bids": [
            [62000.0, 1.5],
            [61999.0, 2.1],
            [61998.0, 0.8],
            [61997.0, 1.2],
            [61996.0, 0.9],
            [61995.0, 1.7],
            [61994.0, 0.6],
            [61993.0, 1.3],
        ],
        "asks": [
            [62001.0, 1.8],
            [62002.0, 2.3],
            [62003.0, 1.1],
            [62004.0, 1.6],
            [62005.0, 0.7],
            [62006.0, 1.4],
            [62007.0, 0.9],
            [62008.0, 1.5],
        ]
    }
    
    # Vectorize order book
    vector = vectorize_order_book(sample_order_book, bit_depth=16)
    
    print(f"Vectorized order book (16-bit): {vector}")
    print(f"Vector shape: {vector.shape}")
    print(f"Vector dtype: {vector.dtype}")
    
    # Compute metrics
    metrics = order_book_vectorizer.compute_vector_metrics(vector)
    print(f"Vector metrics: {metrics}")
    
    # Performance summary
    performance = order_book_vectorizer.get_performance_summary()
    print(f"Performance summary: {performance}")

    # Sample order books for batch test
    sample_order_books = [
        {
            "bids": [[62000.0-i, 1.5+i*0.1] for i in range(8)],
            "asks": [[62001.0+i, 1.8-i*0.1] for i in range(8)]
        } for _ in range(3)
    ]
    symbols = [f"BTC/USDC_{i}" for i in range(3)]
    obv = OrderBookVectorizer(enable_gpu_acceleration=True)
    obv.visualize_batch_comparison(sample_order_books, symbols, bit_depth=16) 