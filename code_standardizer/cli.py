"""
Command-line interface for the Code Standardizer package.

This module provides a command-line interface for matching user codes against a standard code library.
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Optional

from .models import StandardCodeLibrary, UserCode
from .matcher import CodeMatcher
from .similarity import MAX_FIELDS_PER_BATCH
from .utils import (
    load_user_codes_from_json,
    load_user_codes_from_csv,
    save_match_results_to_json,
    save_match_results_to_csv
)
from .config import Config


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging configuration.

    Args:
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Match user codes against a standard code library'
    )

    parser.add_argument(
        '--library',
        '-l',
        required=True,
        help='Path to the standard code library JSON file'
    )

    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help='Path to the input file containing user codes (JSON or CSV)'
    )

    parser.add_argument(
        '--output',
        '-o',
        required=True,
        help='Path to save the match results (JSON or CSV)'
    )

    parser.add_argument(
        '--threshold',
        '-t',
        type=float,
        default=0.7,
        help='Similarity threshold for matching (default: 0.7)'
    )

    parser.add_argument(
        '--config',
        '-c',
        help='Path to the configuration file'
    )

    parser.add_argument(
        '--csv-code-col',
        default='code_value',
        help='Name of the column containing code values in CSV input (default: code_value)'
    )

    parser.add_argument(
        '--csv-name-col',
        default='name',
        help='Name of the column containing code names in CSV input (default: name)'
    )

    parser.add_argument(
        '--csv-desc-col',
        default='description',
        help='Name of the column containing code descriptions in CSV input (default: description)'
    )

    parser.add_argument(
        '--max-fields',
        '-m',
        type=int,
        default=MAX_FIELDS_PER_BATCH,
        help=f'Maximum number of fields to process in a single batch (default: {MAX_FIELDS_PER_BATCH})'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def load_user_codes(
    file_path: str,
    code_col: str = 'code_value',
    name_col: str = 'name',
    desc_col: str = 'description'
) -> List[UserCode]:
    """
    Load user codes from a file.

    Args:
        file_path: Path to the file containing user codes
        code_col: Name of the column containing code values in CSV input
        name_col: Name of the column containing code names in CSV input
        desc_col: Name of the column containing code descriptions in CSV input

    Returns:
        List of UserCode objects
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.json':
        return load_user_codes_from_json(file_path)
    elif file_ext == '.csv':
        return load_user_codes_from_csv(
            file_path,
            code_value_col=code_col,
            name_col=name_col,
            description_col=desc_col
        )
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def save_match_results(results: List, file_path: str) -> None:
    """
    Save match results to a file.

    Args:
        results: List of MatchResult objects
        file_path: Path to save the results
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.json':
        save_match_results_to_json(results, file_path)
    elif file_ext == '.csv':
        save_match_results_to_csv(results, file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def main() -> None:
    """Run the command-line interface."""
    args = parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load configuration if provided
    config = Config(args.config) if args.config else Config()

    # Override threshold from command-line argument
    threshold = args.threshold

    try:
        # Load the standard code library
        logger.info(f"Loading standard code library from {args.library}")
        library = StandardCodeLibrary(args.library)
        logger.info(f"Loaded {len(library)} standard codes")

        # Load user codes
        logger.info(f"Loading user codes from {args.input}")
        user_codes = load_user_codes(
            args.input,
            code_col=args.csv_code_col,
            name_col=args.csv_name_col,
            desc_col=args.csv_desc_col
        )
        logger.info(f"Loaded {len(user_codes)} user codes")

        # Create a code matcher
        max_fields = args.max_fields
        matcher = CodeMatcher(library, threshold=threshold, max_fields=max_fields)

        # Match the user codes
        logger.info(f"Matching user codes with threshold {threshold} and max fields {max_fields}")
        results = matcher.match_codes_batch(user_codes)

        # Count matches
        match_count = sum(1 for result in results if result.is_match)
        logger.info(f"Found {match_count} matches out of {len(results)} user codes")

        # Save the results
        logger.info(f"Saving match results to {args.output}")
        save_match_results(results, args.output)
        logger.info("Done")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
