"""
Example usage of the Code Standardizer package.

This script demonstrates how to use the Code Standardizer package to match user codes
against a standard code library.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_standardizer.models import StandardCode, UserCode, StandardCodeLibrary
from code_standardizer.matcher import CodeMatcher
from code_standardizer.utils import create_sample_standard_library, save_match_results_to_csv


def main():
    """Run the example code standardization process."""
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Create a sample standard library
    library_path = data_dir / 'standard_library.json'
    library = create_sample_standard_library(str(library_path))
    print(f"Created sample standard library with {len(library)} codes")
    
    # Create some sample user codes
    user_codes = [
        UserCode(
            code_value="200",
            name="Success",
            description="The request was successful"
        ),
        UserCode(
            code_value="201",
            name="Created Successfully",
            description="The resource was created"
        ),
        UserCode(
            code_value="400",
            name="Invalid Request",
            description="The request was invalid"
        ),
        UserCode(
            code_value="404",
            name="Resource Not Found",
            description="The requested resource could not be found"
        ),
        UserCode(
            code_value="500",
            name="Server Error",
            description="An internal server error occurred"
        ),
        UserCode(
            code_value="999",
            name="Custom Error",
            description="A custom error occurred"
        )
    ]
    
    # Create a code matcher
    matcher = CodeMatcher(library, threshold=0.7)
    
    # Match the user codes
    results = matcher.match_codes_batch(user_codes)
    
    # Print the results
    print("\nMatching Results:")
    for result in results:
        print(result)
    
    # Save the results to a CSV file
    results_path = data_dir / 'match_results.csv'
    save_match_results_to_csv(results, str(results_path))
    print(f"\nSaved match results to {results_path}")
    
    # Demonstrate updating the standard library
    print("\nUpdating standard library...")
    
    # Add a new standard code
    new_code = StandardCode(
        code_value="999",
        name="Custom Error",
        description="A custom error code for special cases",
        category="Custom"
    )
    library.add_code(new_code)
    library.save_to_file(str(library_path))
    
    # Update the matcher with the new library
    matcher.update_library(library)
    
    # Match the user codes again
    results = matcher.match_codes_batch(user_codes)
    
    # Print the updated results
    print("\nUpdated Matching Results:")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
