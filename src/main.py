#!/usr/bin/env python
"""
BigQuery Data Dictionary Search Engine
Main entry point for the application
"""
import argparse
import json
import sys
from config import Config
from bigquery_client import BigQueryClient
from indexer import TableIndexer
from search_engine import DataDictionarySearchEngine


def build_index(force_rebuild: bool = False):
    """
    Build the search index from BigQuery metadata

    Args:
        force_rebuild: Force rebuilding even if cache exists
    """
    # Initialize components
    indexer = TableIndexer()

    # Try loading from cache first
    if not force_rebuild and Config.USE_CACHE:
        print("Attempting to load index from cache...")
        if indexer.load_index():
            print("Index loaded from cache successfully!")
            return indexer

    # Fetch metadata from BigQuery
    print("Fetching table metadata from BigQuery...")
    bq_client = BigQueryClient()
    tables_metadata = bq_client.fetch_table_metadata()

    if not tables_metadata:
        print("Warning: No table metadata found!")
        return None

    # Build index
    indexer.build_index(tables_metadata)

    # Save to cache
    if Config.USE_CACHE:
        indexer.save_index()

    return indexer


def search_tables(query: str, top_k: int = None, output_format: str = 'text'):
    """
    Search for tables matching the query

    Args:
        query: User's search query
        top_k: Number of results to return
        output_format: Output format ('text' or 'json')
    """
    # Build/load index
    indexer = build_index()
    if indexer is None:
        print("Error: Failed to build index")
        return

    # Initialize search engine
    search_engine = DataDictionarySearchEngine(indexer)

    # Perform search
    results = search_engine.search(query, top_k)

    # Output results
    if output_format == 'json':
        print(json.dumps(results, indent=2))
    else:
        search_engine.print_results(results)


def interactive_mode():
    """Run in interactive mode for multiple queries"""
    print("\n" + "="*80)
    print("BigQuery Data Dictionary Search Engine - Interactive Mode")
    print("="*80)
    print("\nBuilding search index...")

    # Build index once
    indexer = build_index()
    if indexer is None:
        print("Error: Failed to build index")
        return

    search_engine = DataDictionarySearchEngine(indexer)

    # Show index stats
    stats = indexer.get_index_stats()
    print(f"\nIndex ready: {stats['num_tables']} tables indexed")
    print(f"Model: {stats['model']}")
    print("\nEnter your questions (type 'exit' to quit)")
    print("-"*80)

    while True:
        try:
            query = input("\nQuestion: ").strip()

            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break

            if not query:
                continue

            # Perform search
            results = search_engine.search(query)
            search_engine.print_results(results)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='BigQuery Data Dictionary Search Engine'
    )

    parser.add_argument(
        'query',
        nargs='?',
        help='Search query (if not provided, starts interactive mode)'
    )

    parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=None,
        help=f'Number of results to return (default: {Config.TOP_K_RESULTS})'
    )

    parser.add_argument(
        '-f', '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )

    parser.add_argument(
        '--rebuild-index',
        action='store_true',
        help='Force rebuild the search index'
    )

    parser.add_argument(
        '--build-index-only',
        action='store_true',
        help='Only build the index without searching'
    )

    args = parser.parse_args()

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease check your .env file or environment variables.")
        sys.exit(1)

    # Handle build-index-only mode
    if args.build_index_only:
        print("Building index...")
        indexer = build_index(force_rebuild=True)
        if indexer:
            stats = indexer.get_index_stats()
            print(f"\nIndex built successfully!")
            print(f"Tables indexed: {stats['num_tables']}")
            print(f"Model: {stats['model']}")
        sys.exit(0)

    # Handle search query or interactive mode
    if args.query:
        # Single query mode
        search_tables(args.query, args.top_k, args.format)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == '__main__':
    main()
