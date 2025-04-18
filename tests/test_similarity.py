"""
Tests for the similarity module.

This module contains tests for the similarity calculation functions.
"""

import unittest
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_standardizer.models import StandardCode, UserCode
from code_standardizer.similarity import (
    normalize_text,
    levenshtein_distance,
    levenshtein_similarity,
    sequence_matcher_similarity,
    jaccard_similarity,
    calculate_similarity,
    get_best_match
)


class TestNormalizeText(unittest.TestCase):
    """Tests for the normalize_text function."""

    def test_normalize_text_lowercase(self):
        """Test that text is converted to lowercase."""
        self.assertEqual(normalize_text("HELLO WORLD"), "hello world")

    def test_normalize_text_special_chars(self):
        """Test that special characters are removed."""
        self.assertEqual(normalize_text("Hello, World!"), "hello world")

    def test_normalize_text_whitespace(self):
        """Test that extra whitespace is removed."""
        self.assertEqual(normalize_text("  Hello   World  "), "hello world")

    def test_normalize_text_empty(self):
        """Test that empty text remains empty."""
        self.assertEqual(normalize_text(""), "")

    def test_normalize_text_none(self):
        """Test that None is handled correctly."""
        self.assertEqual(normalize_text(None), "")


class TestLevenshteinDistance(unittest.TestCase):
    """Tests for the levenshtein_distance function."""

    def test_levenshtein_distance_identical(self):
        """Test distance between identical strings."""
        self.assertEqual(levenshtein_distance("hello", "hello"), 0)

    def test_levenshtein_distance_different(self):
        """Test distance between different strings."""
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)

    def test_levenshtein_distance_empty(self):
        """Test distance with empty string."""
        self.assertEqual(levenshtein_distance("hello", ""), 5)
        self.assertEqual(levenshtein_distance("", "hello"), 5)

    def test_levenshtein_distance_both_empty(self):
        """Test distance between two empty strings."""
        self.assertEqual(levenshtein_distance("", ""), 0)


class TestSimilarityFunctions(unittest.TestCase):
    """Tests for the similarity functions."""

    def test_levenshtein_similarity(self):
        """Test Levenshtein similarity."""
        self.assertEqual(levenshtein_similarity("hello", "hello"), 1.0)
        self.assertAlmostEqual(levenshtein_similarity("hello", "helo"), 0.8, places=1)
        self.assertEqual(levenshtein_similarity("", ""), 1.0)
        self.assertEqual(levenshtein_similarity("hello", ""), 0.0)

    def test_sequence_matcher_similarity(self):
        """Test SequenceMatcher similarity."""
        self.assertEqual(sequence_matcher_similarity("hello", "hello"), 1.0)
        self.assertGreater(sequence_matcher_similarity("hello", "helo"), 0.8)
        self.assertEqual(sequence_matcher_similarity("", ""), 1.0)
        self.assertEqual(sequence_matcher_similarity("hello", ""), 0.0)

    def test_jaccard_similarity(self):
        """Test Jaccard similarity."""
        self.assertEqual(jaccard_similarity("hello world", "hello world"), 1.0)
        self.assertEqual(jaccard_similarity("hello world", "hello"), 0.5)
        self.assertEqual(jaccard_similarity("", ""), 1.0)
        self.assertEqual(jaccard_similarity("hello", ""), 0.0)


class TestCalculateSimilarity(unittest.TestCase):
    """Tests for the calculate_similarity function."""

    def test_calculate_similarity_identical(self):
        """Test similarity between identical codes."""
        user_code = UserCode(
            code_value="200",
            name="OK",
            description="Request was successful"
        )
        standard_code = StandardCode(
            code_value="200",
            name="OK",
            description="Request was successful",
            category="Success"
        )

        similarity = calculate_similarity(user_code, standard_code)

        self.assertAlmostEqual(similarity['name_similarity'], 1.0, places=1)
        self.assertAlmostEqual(similarity['description_similarity'], 1.0, places=1)
        self.assertAlmostEqual(similarity['overall_similarity'], 1.0, places=1)

    def test_calculate_similarity_similar(self):
        """Test similarity between similar codes."""
        user_code = UserCode(
            code_value="200",
            name="Success",
            description="The request was successful"
        )
        standard_code = StandardCode(
            code_value="200",
            name="OK",
            description="Request was successful",
            category="Success"
        )

        similarity = calculate_similarity(user_code, standard_code)

        self.assertGreaterEqual(similarity['name_similarity'], 0.3)
        self.assertGreaterEqual(similarity['description_similarity'], 0.7)
        self.assertGreaterEqual(similarity['overall_similarity'], 0.5)

    def test_calculate_similarity_different(self):
        """Test similarity between different codes."""
        user_code = UserCode(
            code_value="404",
            name="Not Found",
            description="Resource not found"
        )
        standard_code = StandardCode(
            code_value="200",
            name="OK",
            description="Request was successful",
            category="Success"
        )

        similarity = calculate_similarity(user_code, standard_code)

        self.assertLess(similarity['name_similarity'], 0.3)
        self.assertLess(similarity['description_similarity'], 0.3)
        self.assertLess(similarity['overall_similarity'], 0.3)

    def test_calculate_similarity_no_description(self):
        """Test similarity when user code has no description."""
        user_code = UserCode(
            code_value="200",
            name="OK"
        )
        standard_code = StandardCode(
            code_value="200",
            name="OK",
            description="Request was successful",
            category="Success"
        )

        similarity = calculate_similarity(user_code, standard_code)

        self.assertAlmostEqual(similarity['name_similarity'], 1.0, places=1)
        self.assertEqual(similarity['description_similarity'], 0.0)
        self.assertAlmostEqual(similarity['overall_similarity'], 0.7, places=1)


class TestGetBestMatch(unittest.TestCase):
    """Tests for the get_best_match function."""

    def setUp(self):
        """Set up test data."""
        self.standard_library = [
            StandardCode(
                code_value="200",
                name="OK",
                description="Request was successful",
                category="Success"
            ),
            StandardCode(
                code_value="201",
                name="Created",
                description="Resource was successfully created",
                category="Success"
            ),
            StandardCode(
                code_value="400",
                name="Bad Request",
                description="The request was invalid or cannot be served",
                category="Client Error"
            ),
            StandardCode(
                code_value="404",
                name="Not Found",
                description="The requested resource could not be found",
                category="Client Error"
            )
        ]

    def test_get_best_match_exact(self):
        """Test finding an exact match."""
        user_code = UserCode(
            code_value="200",
            name="OK",
            description="Request was successful"
        )

        best_match, score, above_threshold = get_best_match(
            user_code,
            self.standard_library,
            threshold=0.7
        )

        self.assertEqual(best_match.code_value, "200")
        self.assertAlmostEqual(score, 1.0, places=1)
        self.assertTrue(above_threshold)

    def test_get_best_match_similar(self):
        """Test finding a similar match."""
        user_code = UserCode(
            code_value="404",
            name="Resource Not Found",
            description="The requested resource could not be found"
        )

        best_match, score, above_threshold = get_best_match(
            user_code,
            self.standard_library,
            threshold=0.7
        )

        self.assertEqual(best_match.code_value, "404")
        self.assertGreater(score, 0.7)
        self.assertTrue(above_threshold)

    def test_get_best_match_no_match(self):
        """Test when there is no good match."""
        user_code = UserCode(
            code_value="999",
            name="Custom Error",
            description="A custom error occurred"
        )

        best_match, score, above_threshold = get_best_match(
            user_code,
            self.standard_library,
            threshold=0.7
        )

        self.assertIsNotNone(best_match)  # Should still return the best match
        self.assertLess(score, 0.7)
        self.assertFalse(above_threshold)

    def test_get_best_match_empty_library(self):
        """Test with an empty standard library."""
        user_code = UserCode(
            code_value="200",
            name="OK",
            description="Request was successful"
        )

        best_match, score, above_threshold = get_best_match(
            user_code,
            [],
            threshold=0.7
        )

        self.assertIsNone(best_match)
        self.assertEqual(score, 0.0)
        self.assertFalse(above_threshold)


if __name__ == '__main__':
    unittest.main()
