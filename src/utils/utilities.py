"""
Contains utility functions for Samsung Smart Support Copilot.
"""

import re
from datetime import datetime
from typing import List, Dict, Optional
from system_config import QUERY_TYPES, ALLOWED_FILE_TYPES

def show_welcome_message() -> str:
    """ Welcome message for chat GUI interface."""
    
    message = r"""
    
    Hey, I am here to help you with the following, 
    - 🛠️ **Troubleshooting** : Device problems, errors and issues
    - 🧮 **Product Comparisons** : Comparing Samsung products and features
    - ❓ **General Questions** : Help and guide with the information for general queries

    
    **Do the following to get started:**
    - Upload documents (TXT, PDF or DOCX) using the sidebar
    - Ask your questions in the chat window below

    
    **📝 Example Queries:**
    - "Why is my Galaxy phone overheating?"
    - "How to fix Samsung TV, its not truning on?"
    - "Compare Samsung Galaxy S25 vs S26 features"
    - "How to reset Samsung Smart Washin Machine?"
    - "What if this issue keeps happening again and again ?"(you can ask follow-up questions as well)

    """

    return message

def get_query_type_icon(query_type: str) -> str:
    """ Get the relevant icon for query type"""
    icons = {
        QUERY_TYPES["TROUBLESHOOTING"]: "🛠️",
        QUERY_TYPES["COMPARISON"]: "🧮",
        QUERY_TYPES["GENERAL"]: "❓",
        "follow_up": "🔄"
    }
    return icons.get(query_type.lower(), "❓")

def format_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """ FOrmat a markdown table """
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
    
    # Build header
    header_line = "| "+" | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"

    # Build rows
    row_lines = []
    for row in rows:
        row_line = "| " + " | ".join(
            cell.ljust(col_widths[i]) if i < len(cell) else ""
            for i, cell in enumerate(row)
        ) + " |"
        row_lines.append(row_line)

    return "\n".join([header_line, separator] + row_lines)

def validate_file_type(filename: str, allowed_types: List[str] = None) -> bool:
    """ Validate file type against allowed extensions """
    
    ext = "." + filename.rsplit(".", 1)[-1].lower()
    return ext in ALLOWED_FILE_TYPES

def extract_product_names(query: str) -> List[str]:
    """ Extract product names from comparison query """
    patterns = [
        r"Galaxy\s+\w+\s*\d+",
        r"Samsung\s+\w+",
        r"\w+\s+\d+(?:\s+\w+)?"
    ]

    products = []
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        products.extend(matches)

    # Remove duplicates while preserving order
    seen = set()
    unique_products = []
    for product in products:
        product_lower = product.lower()
        if product_lower not in seen:
            seen.add(product_lower)
            unique_products.append(product)
    
    return unique_products

def format_file_size(size_bytes: int) -> str:
    """ format file size """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def get_timestamp() -> str:
    """ get current time stamp"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_clean_response(response: str) -> str:
    """ Clean the response string from LLM """
    response = response.strip()

    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]

    return response.strip()

def truncate_text_with_ellipsis(text: str, max_length: int = 100) -> str:
    """ truncate text with ellipsis """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def get_confidence_level_of_response(score: float) -> Dict[str, str]:
    """ this returns confidence score description based on score"""
    if score >= 0.8:
        return {"level" : "High", "color": "green", "icon": "👍"}
    elif score >= 0.6:
        return {"level" : "Medium", "color": "orange", "icon": "⚠️"}
    else:
        return {"level" : "Low", "color": "red", "icon": "👎"}
    
def format_sources_display(sources: List[Dict]) -> str:
    """ Format sources section for GUI """
    if not sources:
        return "📚 Source: General knowledge base source."
    
    unique_sources = {}
    for source in sources:
        name = source.get("source", "Unknown")
        page = source.get("page", "N/A")
        if name not in unique_sources:
            unique_sources[name] = []
        if page != "N/A" and page not in unique_sources[name]:
            unique_sources[name].append(page)

    formatted = []
    for name, pages in unique_sources.items():
        if pages:
            formatted.append(f"{name} (Page(s): {", ".join(map(str, pages))})")
        else:
            formatted.append(name)
    
    return f"📚 Sources: {", ".join(formatted)}"

