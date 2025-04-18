# Code Standardizer Optimization

This document describes the optimizations made to the Code Standardizer package to meet specific requirements.

## Optimization Requirements

1. **Semantic Matching**: Use sentence-BERT algorithm for semantic matching of field values
2. **Field Limit**: Limit the number of fields that can be matched at once
3. **Sequential Processing**: Remove parallel processing and implement sequential matching

## Implementation Details

### 1. Semantic Matching with sentence-BERT

We've integrated the `sentence-transformers` library to provide semantic matching capabilities:

- Added a new `sentence_bert_similarity` function that calculates semantic similarity between strings
- Updated the `calculate_similarity` function to use sentence-BERT with higher weight (50%)
- Implemented fallback mechanisms in case the model fails to load or process

The semantic matching provides better results for fields that have similar meanings but different wording.

### 2. Field Limit Implementation

To prevent processing too many fields at once:

- Added a `MAX_FIELDS_PER_BATCH` constant (default: 100) to limit the number of fields
- Updated the `get_best_match` function to limit the number of standard codes processed
- Updated the `match_codes_batch` method to limit the number of user codes processed
- Added a command-line parameter `--max-fields` to control this limit

### 3. Sequential Processing

Removed parallel processing to simplify the implementation:

- Removed the `ThreadPoolExecutor` and related code
- Updated the `match_codes_batch` method to use a simple loop
- Removed the `max_workers` parameter and replaced it with `max_fields`

## Usage Changes

### Command-line Interface

The command-line interface now includes a new parameter:

```bash
python standardize_codes.py --library data/standard_library.json --input data/user_codes.json --output data/match_results.json --threshold 0.6 --max-fields 50
```

### API Usage

When using the library programmatically, you can now specify the maximum number of fields:

```python
from code_standardizer.models import StandardCodeLibrary
from code_standardizer.matcher import CodeMatcher

# Create a standard library
library = StandardCodeLibrary("standard_library.json")

# Create a code matcher with a field limit
matcher = CodeMatcher(library, threshold=0.7, max_fields=50)
```

## Dependencies

The following dependencies have been added:

- `torch`: Required for sentence-transformers
- `sentence-transformers`: For semantic similarity using BERT

You can install these dependencies using:

```bash
pip install -r requirements.txt
```

## Performance Considerations

- The sentence-BERT model is loaded lazily to avoid unnecessary memory usage
- A smaller model (`paraphrase-MiniLM-L6-v2`) is used by default for faster inference
- The field limit helps prevent memory issues when processing large datasets
- Sequential processing simplifies the code and makes it more predictable
