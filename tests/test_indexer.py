"""
Unit tests for indexer
"""
import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from indexer import TableIndexer


@pytest.fixture
def sample_tables():
    """Sample table metadata"""
    return [
        {
            'table_name': 'users',
            'description': 'User account information',
            'columns': [
                {'column_name': 'user_id', 'data_type': 'STRING', 'description': 'User ID'},
                {'column_name': 'email', 'data_type': 'STRING', 'description': 'Email'}
            ]
        },
        {
            'table_name': 'orders',
            'description': 'Order transactions',
            'columns': [
                {'column_name': 'order_id', 'data_type': 'STRING', 'description': 'Order ID'}
            ]
        }
    ]


def test_indexer_initialization():
    """Test indexer initialization"""
    indexer = TableIndexer(model_name='all-MiniLM-L6-v2')
    assert indexer.model_name == 'all-MiniLM-L6-v2'
    assert indexer.embeddings is None
    assert indexer.metadata is None


def test_build_index(sample_tables):
    """Test building index"""
    indexer = TableIndexer(model_name='all-MiniLM-L6-v2')
    indexer.build_index(sample_tables)

    assert indexer.embeddings is not None
    assert indexer.metadata is not None
    assert len(indexer.metadata) == 2
    assert indexer.embeddings.shape[0] == 2


def test_create_searchable_text(sample_tables):
    """Test searchable text creation"""
    indexer = TableIndexer()
    text = indexer._create_searchable_text(sample_tables[0])

    # Should contain table name, description, and column info
    assert 'users' in text.lower()
    assert 'user account' in text.lower()
    assert 'user_id' in text.lower()
    assert 'email' in text.lower()


def test_get_index_stats_not_built():
    """Test index stats when not built"""
    indexer = TableIndexer()
    stats = indexer.get_index_stats()

    assert stats['status'] == 'not_built'


def test_get_index_stats_built(sample_tables):
    """Test index stats when built"""
    indexer = TableIndexer(model_name='all-MiniLM-L6-v2')
    indexer.build_index(sample_tables)
    stats = indexer.get_index_stats()

    assert stats['status'] == 'ready'
    assert stats['num_tables'] == 2
    assert stats['model'] == 'all-MiniLM-L6-v2'
    assert 'embedding_dim' in stats


def test_save_and_load_index(sample_tables, tmp_path):
    """Test saving and loading index"""
    indexer = TableIndexer(model_name='all-MiniLM-L6-v2')
    indexer.build_index(sample_tables)

    # Save index
    cache_path = os.path.join(tmp_path, 'test_index.pkl')
    indexer.save_index(cache_path)

    assert os.path.exists(cache_path)

    # Load index in new indexer
    new_indexer = TableIndexer(model_name='all-MiniLM-L6-v2')
    success = new_indexer.load_index(cache_path)

    assert success
    assert new_indexer.embeddings is not None
    assert len(new_indexer.metadata) == 2


def test_load_nonexistent_index():
    """Test loading non-existent index"""
    indexer = TableIndexer()
    success = indexer.load_index('/nonexistent/path.pkl')

    assert not success
