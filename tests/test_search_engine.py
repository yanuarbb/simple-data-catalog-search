"""
Unit tests for search engine
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from search_engine import DataDictionarySearchEngine
from indexer import TableIndexer


@pytest.fixture
def mock_tables():
    """Mock table metadata"""
    return [
        {
            'table_id': 'project.schema.campaign_data',
            'table_name': 'campaign_data',
            'table_schema': 'schema',
            'description': 'Marketing campaign information',
            'columns': [
                {'column_name': 'campaign_id', 'data_type': 'STRING', 'description': 'Campaign ID'},
                {'column_name': 'budget', 'data_type': 'FLOAT64', 'description': 'Campaign budget'}
            ]
        },
        {
            'table_id': 'project.schema.customer_info',
            'table_name': 'customer_info',
            'table_schema': 'schema',
            'description': 'Customer profile data',
            'columns': [
                {'column_name': 'customer_id', 'data_type': 'STRING', 'description': 'Customer ID'},
                {'column_name': 'email', 'data_type': 'STRING', 'description': 'Email address'}
            ]
        }
    ]


@pytest.fixture
def indexer_with_data(mock_tables):
    """Create indexer with mock data"""
    indexer = TableIndexer(model_name='all-MiniLM-L6-v2')
    indexer.build_index(mock_tables)
    return indexer


def test_search_engine_initialization(indexer_with_data):
    """Test search engine initialization"""
    engine = DataDictionarySearchEngine(indexer_with_data)
    assert engine.indexer is not None
    assert engine.model is not None


def test_search_returns_results(indexer_with_data):
    """Test that search returns results"""
    engine = DataDictionarySearchEngine(indexer_with_data)
    results = engine.search("campaign data", top_k=2)

    assert len(results) == 2
    assert all('table_name' in r for r in results)
    assert all('relevance_score' in r for r in results)
    assert all('rank' in r for r in results)


def test_search_relevance_ranking(indexer_with_data):
    """Test that search ranks by relevance"""
    engine = DataDictionarySearchEngine(indexer_with_data)
    results = engine.search("campaign", top_k=2)

    # First result should be more relevant to "campaign"
    assert results[0]['rank'] == 1
    assert results[1]['rank'] == 2
    # Scores should be in descending order
    assert results[0]['relevance_score'] >= results[1]['relevance_score']


def test_search_with_no_index_raises_error():
    """Test that search without index raises error"""
    indexer = TableIndexer()
    engine = DataDictionarySearchEngine(indexer)

    with pytest.raises(ValueError, match="Index not built"):
        engine.search("test query")


def test_cosine_similarity_calculation(indexer_with_data):
    """Test cosine similarity calculation"""
    engine = DataDictionarySearchEngine(indexer_with_data)

    # Create test vectors
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    similarities = engine._calculate_cosine_similarity(vec1, vec2)

    # First vector should have similarity of 1.0 (identical)
    # Second vector should have similarity of 0.0 (orthogonal)
    assert abs(similarities[0] - 1.0) < 0.001
    assert abs(similarities[1] - 0.0) < 0.001


def test_format_columns():
    """Test column formatting"""
    indexer = TableIndexer()
    engine = DataDictionarySearchEngine(indexer)

    columns = [
        {'column_name': f'col{i}', 'data_type': 'STRING', 'description': f'desc{i}'}
        for i in range(10)
    ]

    formatted = engine._format_columns(columns)

    # Should limit to 5 columns + ellipsis
    assert len(formatted) == 6
    assert formatted[-1]['name'] == '...'
    assert '5 more columns' in formatted[-1]['type']


def test_search_with_custom_top_k(indexer_with_data):
    """Test search with custom top_k parameter"""
    engine = DataDictionarySearchEngine(indexer_with_data)
    results = engine.search("data", top_k=1)

    assert len(results) == 1
