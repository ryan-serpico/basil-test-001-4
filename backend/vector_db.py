"""Vector database service using Polars + NumPy for semantic search"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import polars as pl
import numpy as np

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Vector database service using Polars + NumPy for semantic search"""

    def __init__(self):
        """Initialize vector database"""
        self.df = None
        self.embeddings_matrix = None
        self.is_loaded = False
        self.embedding_dimension = 0

        # Configuration
        self.data_dir = Path("./data")
        self.embeddings_file = self.data_dir / "embeddings.parquet"

    async def initialize(self):
        """Initialize database and load embeddings from Parquet file"""
        try:
            # Load embeddings from Parquet file
            await self._load_embeddings()

            if self.df is not None and len(self.df) > 0:
                # Extract embeddings matrix for fast similarity search
                self._prepare_embeddings_matrix()
                self.is_loaded = True
                logger.info(f"Vector database initialized with {len(self.df)} chunks")
            else:
                logger.warning("No embeddings data found")

        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise

    async def _load_embeddings(self):
        """Load embeddings data from Parquet file"""
        try:
            if not self.embeddings_file.exists():
                raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")

            # Load Parquet file with Polars
            self.df = pl.read_parquet(self.embeddings_file)

            # Validate required columns
            required_columns = ['chunk_id', 'text_content', 'source_document', 'embedding']
            missing_columns = [col for col in required_columns if col not in self.df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns in Parquet file: {missing_columns}")

            # Get embedding dimension from first embedding
            if len(self.df) > 0:
                sample_embedding = self.df['embedding'][0]
                self.embedding_dimension = len(sample_embedding)

            logger.info(f"Loaded {len(self.df)} chunks from Parquet file")
            logger.info(f"Embedding dimension: {self.embedding_dimension}")

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise

    def _prepare_embeddings_matrix(self):
        """Prepare embeddings matrix for fast similarity search using zero-copy operations"""
        try:
            # Extract embeddings using Polars to_numpy() for zero-copy operation
            # This converts the List[Float32] column to a 2D numpy array efficiently
            embeddings_list = self.df['embedding'].to_list()

            # Convert to numpy matrix (should be zero-copy with proper conditions)
            self.embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

            logger.info(f"Prepared embeddings matrix: {self.embeddings_matrix.shape}")

            # Verify embeddings are unit normalized (they should be from OpenAI)
            norms = np.linalg.norm(self.embeddings_matrix, axis=1)
            avg_norm = np.mean(norms)
            logger.info(f"Average embedding norm: {avg_norm:.4f} (should be ~1.0 for normalized embeddings)")

        except Exception as e:
            logger.error(f"Failed to prepare embeddings matrix: {e}")
            raise

    def _fast_similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast similarity search using numpy dot products

        Based on Max Woolf's approach: for unit-normalized embeddings,
        dot product equals cosine similarity

        Args:
            query_embedding: Query embedding vector (should be normalized)
            k: Number of top results to return

        Returns:
            Tuple of (indices, similarity_scores)
        """
        # Ensure query is normalized
        if np.linalg.norm(query_embedding) > 0:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Calculate dot products (cosine similarity for normalized vectors)
        dot_products = query_embedding @ self.embeddings_matrix.T

        # Find top k most similar using argpartition (faster than full sort)
        if k >= len(dot_products):
            # Return all results
            idx = np.argsort(dot_products)[::-1]
            scores = dot_products[idx]
        else:
            # Use argpartition for efficiency
            idx = np.argpartition(dot_products, -k)[-k:]
            idx = idx[np.argsort(dot_products[idx])[::-1]]
            scores = dot_products[idx]

        return idx, scores

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.0,
        source_document_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with optional filtering

        Args:
            query: Search query text
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold (0-1)
            source_document_filter: Optional document name to filter by

        Returns:
            List of search results with similarity scores
        """
        try:
            if not self.is_loaded or self.df is None or self.embeddings_matrix is None:
                raise RuntimeError("Vector database not initialized")

            # For this template, we need to generate query embedding
            # In a real implementation, this would use the same embedding model
            # For now, we'll create a dummy embedding and show the pattern
            logger.warning("Query embedding generation not implemented in template - using dummy embedding")
            query_embedding = np.random.randn(self.embedding_dimension).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

            # Apply document filter if specified
            if source_document_filter:
                filtered_df = self.df.filter(
                    pl.col("source_document").str.contains(source_document_filter)
                )

                if len(filtered_df) == 0:
                    logger.info(f"No documents found matching filter: {source_document_filter}")
                    return []

                # Get filtered embeddings matrix
                filtered_embeddings = np.array(filtered_df['embedding'].to_list(), dtype=np.float32)

                # Perform search on filtered data
                indices, scores = self._fast_similarity_search(query_embedding, k=limit)

                # Map back to original indices
                results_df = filtered_df[indices].with_columns(
                    pl.Series("similarity_score", scores)
                ).filter(pl.col("similarity_score") >= min_similarity)

            else:
                # Perform search on full dataset
                indices, scores = self._fast_similarity_search(query_embedding, k=limit)

                # Create results dataframe
                results_df = self.df[indices].with_columns(
                    pl.Series("similarity_score", scores)
                ).filter(pl.col("similarity_score") >= min_similarity)

            # Format results
            formatted_results = []

            for row in results_df.iter_rows(named=True):
                # Get context (surrounding chunks)
                context_before, context_after = await self._get_context(
                    row['source_document'],
                    row['chunk_position']
                )

                result = {
                    "text": row["text_content"],
                    "metadata": {
                        "source_document": row["source_document"],
                        "chunk_position": row["chunk_position"],
                        "token_count": row.get("token_count", 0),
                        "context_before": context_before,
                        "context_after": context_after
                    },
                    "similarity_score": round(float(row["similarity_score"]), 4)
                }
                formatted_results.append(result)

            logger.info(f"Found {len(formatted_results)} results for query (limit: {limit})")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def _get_context(
        self,
        source_document: str,
        chunk_position: int
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get context chunks (before and after current chunk)"""
        try:
            doc_chunks = self.df.filter(
                pl.col("source_document") == source_document
            ).sort("chunk_position")

            context_before = None
            context_after = None

            # Find chunks with position-1 and position+1
            before_chunk = doc_chunks.filter(
                pl.col("chunk_position") == chunk_position - 1
            )
            after_chunk = doc_chunks.filter(
                pl.col("chunk_position") == chunk_position + 1
            )

            if len(before_chunk) > 0:
                context_before = before_chunk["text_content"][0][:200]  # Truncate for brevity

            if len(after_chunk) > 0:
                context_after = after_chunk["text_content"][0][:200]  # Truncate for brevity

            return context_before, context_after

        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return None, None

    async def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of all documents in the collection"""
        try:
            if not self.is_loaded or self.df is None:
                return []

            # Group by document and calculate statistics
            doc_stats = (
                self.df.group_by("source_document")
                .agg([
                    pl.len().alias("chunks"),
                    pl.col("token_count").sum().alias("tokens")
                ])
                .sort("source_document")
            )

            documents = []
            for row in doc_stats.iter_rows(named=True):
                documents.append({
                    "name": row["source_document"],
                    "chunks": row["chunks"],
                    "tokens": row["tokens"],
                    "size_bytes": 0  # Not available in Parquet format
                })

            return documents

        except Exception as e:
            logger.error(f"Failed to get document list: {e}")
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            if not self.is_loaded or self.df is None:
                return {"error": "Database not loaded"}

            total_chunks = len(self.df)
            unique_documents = self.df["source_document"].n_unique()
            total_tokens = self.df["token_count"].sum() if "token_count" in self.df.columns else 0

            return {
                "total_documents": unique_documents,
                "total_chunks": total_chunks,
                "total_tokens": int(total_tokens),
                "embedding_dimension": self.embedding_dimension,
                "average_chunks_per_document": round(total_chunks / unique_documents, 1) if unique_documents > 0 else 0,
                "collection_loaded": self.is_loaded,
                "parquet_file_size_mb": round(self.embeddings_file.stat().st_size / (1024 * 1024), 2) if self.embeddings_file.exists() else 0,
                "storage_format": "Parquet + Polars",
                "search_method": "NumPy dot product"
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}

    async def export_data(
        self,
        format: str = "parquet",
        query: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Export data in various formats

        Args:
            format: Export format ('parquet', 'csv', 'json')
            query: Optional search query to filter results
            limit: Optional limit on number of records

        Returns:
            Export information and data
        """
        try:
            if not self.is_loaded or self.df is None:
                raise RuntimeError("Database not loaded")

            # Start with full dataset or search results
            if query:
                # Would normally search first, but for template we'll use full data
                export_df = self.df
                logger.warning("Query-based export not fully implemented in template")
            else:
                export_df = self.df

            # Apply limit if specified
            if limit and limit > 0:
                export_df = export_df.head(limit)

            if format.lower() == "parquet":
                # Already in Parquet format - just copy file info
                return {
                    "format": "parquet",
                    "records": len(export_df),
                    "file_info": "Data available in native Parquet format",
                    "columns": export_df.columns
                }

            elif format.lower() == "csv":
                # Convert to CSV (excluding embeddings for size)
                csv_df = export_df.select([
                    col for col in export_df.columns if col != "embedding"
                ])
                csv_data = csv_df.write_csv()

                return {
                    "format": "csv",
                    "records": len(csv_df),
                    "data": csv_data,
                    "note": "Embeddings excluded from CSV export due to size"
                }

            elif format.lower() == "json":
                # Convert to JSON (excluding embeddings for size)
                json_df = export_df.select([
                    col for col in export_df.columns if col != "embedding"
                ])
                json_data = json_df.write_json()

                return {
                    "format": "json",
                    "records": len(json_df),
                    "data": json_data,
                    "note": "Embeddings excluded from JSON export due to size"
                }
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {"error": str(e)}

    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.utcnow().isoformat()

    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            health_status = {
                "database_loaded": self.is_loaded,
                "parquet_file_exists": self.embeddings_file.exists(),
                "data_directory_exists": self.data_dir.exists(),
                "embeddings_matrix_loaded": self.embeddings_matrix is not None,
                "embedding_dimension": self.embedding_dimension
            }

            if self.df is not None:
                health_status["total_chunks"] = len(self.df)
                health_status["unique_documents"] = self.df["source_document"].n_unique()

            # Overall health check
            health_status["overall_healthy"] = all([
                health_status["database_loaded"],
                health_status["parquet_file_exists"],
                health_status["embeddings_matrix_loaded"]
            ])

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "overall_healthy": False,
                "error": str(e)
            }