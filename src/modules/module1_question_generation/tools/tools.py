import json
import re
import sqlparse

def verify_json_format(text):
    """Check if the text is a valid JSON"""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def verify_sql_query(text):
    """Check if the text is a valid SQL query using sqlparse"""
    try:
        parsed = sqlparse.parse(text)
        if not parsed:
            return False
        # Basic validation: Check for common SQL commands
        tokens = [token.ttype for token in parsed[0].tokens if not token.is_whitespace]
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]
        return any(keyword in text.upper() for keyword in sql_keywords)
    except Exception:
        return False

def verify_regex(text, pattern):
    """Check if the text matches the given regex pattern"""
    try:
        return bool(re.search(pattern, text))
    except re.error:
        return False  # Invalid regex pattern

def verify_contains(text, substring):
    """Check if the text contains the given substring (case-insensitive)"""
    return substring.lower() in text.lower()
