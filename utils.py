import re

def clean_query(query):
    """Cleans up the input query to ensure safe handling."""
    return re.sub(r"[;'\"]", "", query)
