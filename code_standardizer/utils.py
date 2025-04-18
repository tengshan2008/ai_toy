"""
Utility functions for the Code Standardizer package.

This module provides helper functions for working with code standardization.
"""

import json
import csv
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from .models import StandardCode, UserCode, StandardCodeLibrary


def load_user_codes_from_json(file_path: str) -> List[UserCode]:
    """
    Load user codes from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of UserCode objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return [UserCode.from_dict(item) for item in data]
    else:
        raise ValueError("JSON file must contain a list of user codes")


def load_user_codes_from_csv(
    file_path: str,
    code_value_col: str = 'code_value',
    name_col: str = 'name',
    description_col: Optional[str] = 'description'
) -> List[UserCode]:
    """
    Load user codes from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        code_value_col: Name of the column containing code values
        name_col: Name of the column containing code names
        description_col: Name of the column containing code descriptions
        
    Returns:
        List of UserCode objects
    """
    user_codes = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Extract required fields
            code_value = row.pop(code_value_col, None)
            name = row.pop(name_col, None)
            
            if code_value is None or name is None:
                continue
            
            # Extract optional description
            description = None
            if description_col and description_col in row:
                description = row.pop(description_col)
            
            # All remaining fields go into additional_fields
            user_code = UserCode(
                code_value=code_value,
                name=name,
                description=description,
                additional_fields=row
            )
            
            user_codes.append(user_code)
    
    return user_codes


def save_match_results_to_json(results: List[Any], file_path: str) -> None:
    """
    Save match results to a JSON file.
    
    Args:
        results: List of MatchResult objects
        file_path: Path to save the JSON file
    """
    data = [result.to_dict() for result in results]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_match_results_to_csv(results: List[Any], file_path: str) -> None:
    """
    Save match results to a CSV file.
    
    Args:
        results: List of MatchResult objects
        file_path: Path to save the CSV file
    """
    if not results:
        return
    
    # Prepare the data
    rows = []
    for result in results:
        row = {
            'user_code_value': result.user_code.code_value,
            'user_code_name': result.user_code.name,
            'similarity_score': result.similarity_score,
            'is_match': result.is_match,
            'threshold': result.threshold
        }
        
        if result.matched_code:
            row.update({
                'matched_code_value': result.matched_code.code_value,
                'matched_code_name': result.matched_code.name,
                'matched_code_description': result.matched_code.description
            })
        else:
            row.update({
                'matched_code_value': '',
                'matched_code_name': '',
                'matched_code_description': ''
            })
        
        rows.append(row)
    
    # Write to CSV
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def create_sample_standard_library(file_path: str) -> StandardCodeLibrary:
    """
    Create a sample standard code library for testing.
    
    Args:
        file_path: Path to save the library
        
    Returns:
        The created StandardCodeLibrary
    """
    library = StandardCodeLibrary()
    
    # Add some sample standard codes
    sample_codes = [
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
            code_value="401",
            name="Unauthorized",
            description="Authentication credentials were missing or incorrect",
            category="Client Error"
        ),
        StandardCode(
            code_value="403",
            name="Forbidden",
            description="The server understood the request but refuses to authorize it",
            category="Client Error"
        ),
        StandardCode(
            code_value="404",
            name="Not Found",
            description="The requested resource could not be found",
            category="Client Error"
        ),
        StandardCode(
            code_value="500",
            name="Internal Server Error",
            description="An unexpected condition was encountered",
            category="Server Error"
        ),
        StandardCode(
            code_value="503",
            name="Service Unavailable",
            description="The server is currently unavailable",
            category="Server Error"
        )
    ]
    
    for code in sample_codes:
        library.add_code(code)
    
    # Save the library to file
    library.save_to_file(file_path)
    
    return library
