# Code Standardizer

A Python package for standardizing interface return codes by matching them against a standard code library.

## Overview

The Code Standardizer package provides functionality to match user-uploaded interface return codes against a standard code library. It uses text similarity algorithms to find the best matches and provides recommendations based on configurable similarity thresholds.

## Features

- Match user codes against a standard code library
- Calculate similarity using multiple metrics (Levenshtein, SequenceMatcher, Jaccard)
- Support for batch processing of multiple codes
- Configurable similarity thresholds
- Support for updating the standard library
- Comprehensive test suite

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/code-standardizer.git
cd code-standardizer

# Install the package
pip install -e .
```

## Usage

### Basic Usage

```python
from code_standardizer.models import StandardCode, UserCode, StandardCodeLibrary
from code_standardizer.matcher import CodeMatcher

# Create a standard library
library = StandardCodeLibrary()

# Add standard codes to the library
library.add_code(StandardCode(
    code_value="200",
    name="OK",
    description="Request was successful",
    category="Success"
))

library.add_code(StandardCode(
    code_value="404",
    name="Not Found",
    description="The requested resource could not be found",
    category="Client Error"
))

# Save the library to a file
library.save_to_file("standard_library.json")

# Create a code matcher
matcher = CodeMatcher(library, threshold=0.7)

# Match a single user code
user_code = UserCode(
    code_value="200",
    name="Success",
    description="The request was successful"
)

result = matcher.match_code(user_code)
print(result)  # Prints the match result

# Match multiple user codes
user_codes = [
    UserCode(code_value="200", name="Success", description="The request was successful"),
    UserCode(code_value="404", name="Resource Not Found", description="The resource was not found"),
    UserCode(code_value="999", name="Custom Error", description="A custom error occurred")
]

results = matcher.match_codes_batch(user_codes)
for result in results:
    print(result)
```

### Loading User Codes from Files

```python
from code_standardizer.utils import load_user_codes_from_json, load_user_codes_from_csv

# Load user codes from a JSON file
user_codes = load_user_codes_from_json("user_codes.json")

# Load user codes from a CSV file
user_codes = load_user_codes_from_csv(
    "user_codes.csv",
    code_value_col="code",
    name_col="name",
    description_col="description"
)
```

### Saving Match Results

```python
from code_standardizer.utils import save_match_results_to_json, save_match_results_to_csv

# Save match results to a JSON file
save_match_results_to_json(results, "match_results.json")

# Save match results to a CSV file
save_match_results_to_csv(results, "match_results.csv")
```

### Updating the Standard Library

```python
# Load a new standard library
new_library = StandardCodeLibrary("new_standard_library.json")

# Update the matcher with the new library
matcher.update_library(new_library)
```

## Configuration

You can configure the Code Standardizer using the `Config` class:

```python
from code_standardizer.config import Config

# Load configuration from a file
config = Config("config.json")

# Get a configuration value
threshold = config.get("similarity_threshold")

# Set a configuration value
config.set("similarity_threshold", 0.8)

# Save configuration to a file
config.save_to_file("config.json")
```

## Running Tests

```bash
# Run all tests
python -m unittest discover

# Run specific test modules
python -m unittest tests.test_similarity
python -m unittest tests.test_matcher
```

## Example

See the `examples/example_usage.py` script for a complete example of how to use the Code Standardizer package.