"""
Search engine for finding relevant tables based on user queries
"""
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from indexer import TableIndexer
from config import Config


class DataDictionarySearchEngine:
    """Search engine for BigQuery data dictionary"""

    def __init__(self, indexer: TableIndexer):
        """
        Initialize search engine

        Args:
            indexer: TableIndexer instance with built index
        """
        self.indexer = indexer
        self.model = indexer.model

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for tables matching the query

        Args:
            query: User's natural language question
            top_k: Number of top results to return (default from config)

        Returns:
            List of search results with relevance scores
        """
        if top_k is None:
            top_k = Config.TOP_K_RESULTS

        # Check if index is built
        if self.indexer.embeddings is None or self.indexer.metadata is None:
            raise ValueError("Index not built. Please build index first.")

        print(f"\nSearching for: '{query}'")

        # Encode the query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )

        # Calculate cosine similarities
        similarities = self._calculate_cosine_similarity(
            query_embedding,
            self.indexer.embeddings
        )

        # Get top K results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Prepare results
        results = []
        for i, idx in enumerate(top_indices):
            table = self.indexer.metadata[idx]
            score = float(similarities[idx])

            result = {
                'rank': i + 1,
                'table_id': table.get('table_id', ''),
                'table_name': table.get('table_name', ''),
                'table_schema': table.get('table_schema', ''),
                'description': table.get('description', ''),
                'relevance_score': round(score, 4),
                'columns': self._format_columns(table.get('columns', []))
            }

            results.append(result)

        return results

    def _calculate_cosine_similarity(
        self,
        query_vec: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between query and all embeddings

        Args:
            query_vec: Query embedding vector
            embeddings: Matrix of table embeddings

        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query_vec / np.linalg.norm(query_vec)
        embeddings_norm = embeddings / np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        # Compute cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)

        return similarities

    def _format_columns(self, columns: List[Dict]) -> List[Dict]:
        """
        Format column information for output

        Args:
            columns: List of column metadata

        Returns:
            Formatted column list (limited to first 5 for brevity)
        """
        # Limit to first 5 columns to keep output concise
        limited_columns = columns[:5]

        formatted = []
        for col in limited_columns:
            formatted.append({
                'name': col.get('column_name', ''),
                'type': col.get('data_type', ''),
                'description': col.get('description', '')
            })

        if len(columns) > 5:
            formatted.append({
                'name': '...',
                'type': f'({len(columns) - 5} more columns)',
                'description': ''
            })

        return formatted

    def print_results(self, results: List[Dict]) -> None:
        """
        Pretty print search results

        Args:
            results: List of search results
        """
        if not results:
            print("\nNo results found.")
            return

        print(f"\n{'='*80}")
        print(f"Found {len(results)} relevant tables:")
        print(f"{'='*80}\n")

        for result in results:
            print(f"Rank #{result['rank']}")
            print(f"Table: {result['table_name']}")
            print(f"Schema: {result['table_schema']}")
            print(f"Full ID: {result['table_id']}")
            print(f"Relevance Score: {result['relevance_score']}")
            print(f"Description: {result['description']}")

            if result['columns']:
                print(f"\nKey Columns:")
                for col in result['columns']:
                    if col['name'] == '...':
                        print(f"  - {col['type']}")
                    else:
                        col_desc = f" - {col['description']}" if col['description'] else ""
                        print(f"  - {col['name']} ({col['type']}){col_desc}")

            print(f"\n{'-'*80}\n")
