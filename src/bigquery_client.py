"""
BigQuery client for fetching table metadata
"""
from google.cloud import bigquery
from typing import List, Dict
import json
from config import Config


class BigQueryClient:
    """Client for interacting with BigQuery to fetch table metadata"""

    def __init__(self):
        """Initialize BigQuery client"""
        if Config.USE_MOCK_DATA:
            self.client = None
        else:
            if Config.SERVICE_ACCOUNT_KEY_PATH:
                # Validate that the service account key file exists
                import os
                if not os.path.exists(Config.SERVICE_ACCOUNT_KEY_PATH):
                    print(f"Error: Service account key file not found at: {Config.SERVICE_ACCOUNT_KEY_PATH}")
                    print("Please either:")
                    print("  1. Set the correct path in .env file (SERVICE_ACCOUNT_KEY_PATH)")
                    print("  2. Or use mock data by setting USE_MOCK_DATA=true")
                    raise FileNotFoundError(f"Service account key not found: {Config.SERVICE_ACCOUNT_KEY_PATH}")

                self.client = bigquery.Client.from_service_account_json(
                    Config.SERVICE_ACCOUNT_KEY_PATH,
                    project=Config.GCP_PROJECT_ID
                )
            else:
                # Use default credentials
                self.client = bigquery.Client(project=Config.GCP_PROJECT_ID)

    def fetch_table_metadata(self) -> List[Dict[str, str]]:
        """
        Fetch metadata for all tables in the configured dataset

        Returns:
            List of dictionaries containing table metadata
        """
        if Config.USE_MOCK_DATA:
            return self._load_mock_data()

        query = f"""
        SELECT
            table_catalog,
            table_schema,
            table_name,
            IFNULL(option_value, 'No description available') as table_description
        FROM
            `{Config.GCP_PROJECT_ID}.{Config.BIGQUERY_DATASET}.INFORMATION_SCHEMA.TABLE_OPTIONS`
        WHERE
            option_name = 'description'
        ORDER BY
            table_name
        """

        try:
            query_job = self.client.query(query)
            results = query_job.result()

            tables_metadata = []
            for row in results:
                table_id = f"{row.table_catalog}.{row.table_schema}.{row.table_name}"

                # Fetch column information
                columns_info = self._fetch_column_metadata(
                    row.table_catalog,
                    row.table_schema,
                    row.table_name
                )

                tables_metadata.append({
                    'table_id': table_id,
                    'table_name': row.table_name,
                    'table_schema': row.table_schema,
                    'description': row.table_description,
                    'columns': columns_info
                })

            return tables_metadata

        except Exception as e:
            print(f"Error fetching table metadata: {e}")
            raise

    def _fetch_column_metadata(
        self,
        catalog: str,
        schema: str,
        table: str
    ) -> List[Dict[str, str]]:
        """
        Fetch column metadata for a specific table

        Args:
            catalog: Table catalog (project)
            schema: Table schema (dataset)
            table: Table name

        Returns:
            List of column metadata dictionaries
        """
        query = f"""
        SELECT
            column_name,
            data_type,
            IFNULL(description, '') as column_description
        FROM
            `{catalog}.{schema}.INFORMATION_SCHEMA.COLUMN_FIELD_PATHS`
        WHERE
            table_name = '{table}'
        ORDER BY
            ordinal_position
        """

        try:
            query_job = self.client.query(query)
            results = query_job.result()

            columns = []
            for row in results:
                columns.append({
                    'column_name': row.column_name,
                    'data_type': row.data_type,
                    'description': row.column_description
                })

            return columns

        except Exception as e:
            print(f"Error fetching column metadata: {e}")
            return []

    def _load_mock_data(self) -> List[Dict[str, str]]:
        """Load mock data from JSON file for testing"""
        try:
            # Get absolute path to mock data file
            # Works from any directory by resolving relative to this file
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)  # Go up from src/ to project root
            mock_data_path = os.path.join(project_root, 'data', 'mock_tables.json')

            with open(mock_data_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: Mock data file not found. Returning empty list.")
            return []
