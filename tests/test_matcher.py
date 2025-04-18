"""
Tests for the matcher module.

This module contains tests for the code matching functionality.
"""

import unittest
import sys
import tempfile
import json
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_standardizer.models import StandardCode, UserCode, StandardCodeLibrary
from code_standardizer.matcher import CodeMatcher, MatchResult


class TestMatchResult(unittest.TestCase):
    """Tests for the MatchResult class."""

    def test_match_result_init(self):
        """Test MatchResult initialization."""
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

        result = MatchResult(
            user_code=user_code,
            matched_code=standard_code,
            similarity_score=0.9,
            is_match=True,
            threshold=0.7
        )

        self.assertEqual(result.user_code, user_code)
        self.assertEqual(result.matched_code, standard_code)
        self.assertEqual(result.similarity_score, 0.9)
        self.assertTrue(result.is_match)
        self.assertEqual(result.threshold, 0.7)

    def test_match_result_to_dict(self):
        """Test converting MatchResult to dictionary."""
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

        result = MatchResult(
            user_code=user_code,
            matched_code=standard_code,
            similarity_score=0.9,
            is_match=True,
            threshold=0.7
        )

        result_dict = result.to_dict()

        self.assertIn('user_code', result_dict)
        self.assertIn('matched_code', result_dict)
        self.assertEqual(result_dict['similarity_score'], 0.9)
        self.assertTrue(result_dict['is_match'])
        self.assertEqual(result_dict['threshold'], 0.7)

    def test_match_result_str(self):
        """Test string representation of MatchResult."""
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

        # Test with a match
        result = MatchResult(
            user_code=user_code,
            matched_code=standard_code,
            similarity_score=0.9,
            is_match=True,
            threshold=0.7
        )

        self.assertIn("Match found", str(result))
        self.assertIn("0.90", str(result))

        # Test with no match
        result = MatchResult(
            user_code=user_code,
            matched_code=standard_code,
            similarity_score=0.5,
            is_match=False,
            threshold=0.7
        )

        self.assertIn("No match found", str(result))
        self.assertIn("0.50", str(result))


class TestCodeMatcher(unittest.TestCase):
    """Tests for the CodeMatcher class."""

    def setUp(self):
        """Set up test data."""
        # Create a temporary standard library file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.library_path = os.path.join(self.temp_dir.name, 'standard_library.json')

        # Create standard codes
        self.standard_codes = [
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

        # Create a standard library
        self.library = StandardCodeLibrary()
        for code in self.standard_codes:
            self.library.add_code(code)

        # Save the library to file
        self.library.save_to_file(self.library_path)

        # Create user codes
        self.user_codes = [
            UserCode(
                code_value="200",
                name="OK",
                description="Request was successful"
            ),
            UserCode(
                code_value="404",
                name="Resource Not Found",
                description="The requested resource could not be found"
            ),
            UserCode(
                code_value="999",
                name="Custom Error",
                description="A custom error occurred"
            )
        ]

        # Create a code matcher
        self.matcher = CodeMatcher(self.library, threshold=0.7)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_match_code(self):
        """Test matching a single code."""
        # Test with an exact match
        result = self.matcher.match_code(self.user_codes[0])

        self.assertTrue(result.is_match)
        self.assertEqual(result.matched_code.code_value, "200")
        self.assertGreaterEqual(result.similarity_score, 0.9)

        # Test with a similar match
        result = self.matcher.match_code(self.user_codes[1])

        self.assertTrue(result.is_match)
        self.assertEqual(result.matched_code.code_value, "404")
        self.assertGreaterEqual(result.similarity_score, 0.7)

        # Test with no match
        result = self.matcher.match_code(self.user_codes[2])

        self.assertFalse(result.is_match)
        self.assertIsNotNone(result.matched_code)  # Should still return the best match
        self.assertLess(result.similarity_score, 0.7)

    def test_match_codes_batch(self):
        """Test matching multiple codes in batch."""
        results = self.matcher.match_codes_batch(self.user_codes)

        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].is_match)  # First code should match
        self.assertTrue(results[1].is_match)  # Second code should match
        self.assertFalse(results[2].is_match)  # Third code should not match

    def test_update_library(self):
        """Test updating the standard library."""
        # Create a new library with an additional code
        new_library = StandardCodeLibrary()
        for code in self.standard_codes:
            new_library.add_code(code)

        new_code = StandardCode(
            code_value="999",
            name="Custom Error",
            description="A custom error code for special cases",
            category="Custom"
        )
        new_library.add_code(new_code)

        # Update the matcher with the new library
        self.matcher.update_library(new_library)

        # Test matching the previously unmatched code
        result = self.matcher.match_code(self.user_codes[2])

        self.assertTrue(result.is_match)
        self.assertEqual(result.matched_code.code_value, "999")
        self.assertGreaterEqual(result.similarity_score, 0.7)

    def test_set_threshold(self):
        """Test setting the similarity threshold."""
        # Lower the threshold
        self.matcher.set_threshold(0.1)  # Set to a very low threshold

        # Test with a code that previously didn't match
        result = self.matcher.match_code(self.user_codes[2])

        # With a very low threshold, even a poor match should be considered a match
        self.assertTrue(result.is_match)  # Should now match due to lower threshold

        # Raise the threshold
        self.matcher.set_threshold(0.9)

        # Test with a code that previously matched
        result = self.matcher.match_code(self.user_codes[1])

        self.assertFalse(result.is_match)  # Should now not match due to higher threshold

    def test_invalid_threshold(self):
        """Test setting an invalid threshold."""
        with self.assertRaises(ValueError):
            self.matcher.set_threshold(1.5)

        with self.assertRaises(ValueError):
            self.matcher.set_threshold(-0.1)


if __name__ == '__main__':
    unittest.main()
