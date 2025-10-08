"""
Configuration management for BigQuery Data Dictionary Search Engine
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the application"""

    # BigQuery configuration
    GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', '')
    BIGQUERY_DATASET = os.getenv('BIGQUERY_DATASET', '')
    SERVICE_ACCOUNT_KEY_PATH = os.getenv('SERVICE_ACCOUNT_KEY_PATH', '')

    # Search engine configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '10'))

    # Cache configuration
    # Convert to absolute path so it works from any directory
    _cache_dir = os.getenv('CACHE_DIR', 'data/cache')
    if not os.path.isabs(_cache_dir):
        # If relative path, make it relative to project root (parent of src)
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.dirname(_script_dir)
        CACHE_DIR = os.path.join(_project_root, _cache_dir)
    else:
        CACHE_DIR = _cache_dir

    USE_CACHE = os.getenv('USE_CACHE', 'true').lower() == 'true'

    # Mock data mode (for testing without BigQuery access)
    USE_MOCK_DATA = os.getenv('USE_MOCK_DATA', 'false').lower() == 'true'

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.USE_MOCK_DATA:
            if not cls.GCP_PROJECT_ID:
                raise ValueError("GCP_PROJECT_ID is required when not using mock data")
            if not cls.BIGQUERY_DATASET:
                raise ValueError("BIGQUERY_DATASET is required when not using mock data")

        return True
