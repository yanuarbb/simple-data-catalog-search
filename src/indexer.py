"""
Indexer for building search index from table metadata
"""
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from config import Config


class TableIndexer:
    """Build and manage search index for table metadata"""

    def __init__(self, model_name: str = None):
        """
        Initialize the indexer with a sentence transformer model

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embeddings = None
        self.metadata = None

    def build_index(self, tables_metadata: List[Dict]) -> None:
        """
        Build search index from table metadata

        Args:
            tables_metadata: List of table metadata dictionaries
        """
        print(f"Building index for {len(tables_metadata)} tables...")

        # Create searchable text for each table
        searchable_texts = []
        for table in tables_metadata:
            text = self._create_searchable_text(table)
            searchable_texts.append(text)

        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = self.model.encode(
            searchable_texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Store metadata
        self.metadata = tables_metadata

        print(f"Index built successfully with {len(searchable_texts)} tables")

    def _create_searchable_text(self, table: Dict) -> str:
        """
        Create searchable text from table metadata

        This combines table name, description, and column information
        into a single string for embedding

        Args:
            table: Table metadata dictionary

        Returns:
            Searchable text string
        """
        parts = []

        # Add table name (weighted more by repeating)
        table_name = table.get('table_name', '')
        parts.append(f"{table_name} {table_name}")

        # Add description
        description = table.get('description', '')
        if description and description != 'No description available':
            parts.append(description)

        # Add column names and descriptions
        columns = table.get('columns', [])
        column_names = []
        column_descs = []

        for col in columns:
            col_name = col.get('column_name', '')
            if col_name:
                column_names.append(col_name)

            col_desc = col.get('description', '')
            if col_desc:
                column_descs.append(col_desc)

        if column_names:
            parts.append("Columns: " + " ".join(column_names))

        if column_descs:
            parts.append(" ".join(column_descs))

        return " ".join(parts)

    def save_index(self, cache_path: str = None) -> None:
        """
        Save index to disk for faster loading

        Args:
            cache_path: Path to save the index cache
        """
        if not cache_path:
            os.makedirs(Config.CACHE_DIR, exist_ok=True)
            cache_path = os.path.join(Config.CACHE_DIR, 'index.pkl')

        with open(cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'model_name': self.model_name
            }, f)

        print(f"Index saved to {cache_path}")

    def load_index(self, cache_path: str = None) -> bool:
        """
        Load index from disk

        Args:
            cache_path: Path to load the index cache from

        Returns:
            True if successful, False otherwise
        """
        if not cache_path:
            cache_path = os.path.join(Config.CACHE_DIR, 'index.pkl')

        if not os.path.exists(cache_path):
            print(f"Cache file not found at {cache_path}")
            return False

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            # Verify model compatibility
            if data['model_name'] != self.model_name:
                print(f"Warning: Cached model ({data['model_name']}) differs from current model ({self.model_name})")
                return False

            self.embeddings = data['embeddings']
            self.metadata = data['metadata']

            print(f"Index loaded from {cache_path}")
            return True

        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def get_index_stats(self) -> Dict:
        """
        Get statistics about the current index

        Returns:
            Dictionary with index statistics
        """
        if self.embeddings is None or self.metadata is None:
            return {'status': 'not_built'}

        return {
            'status': 'ready',
            'num_tables': len(self.metadata),
            'embedding_dim': self.embeddings.shape[1],
            'model': self.model_name
        }
