"""
Tools Definition and Dispatcher Module
Defines the tool schemas (JSON Schema) provided to the Gemini LLM
and executes the corresponding Python functions in src.retrieval.
"""

from typing import Dict, Any, List
from src.retrieval import search_fda, search_ema, get_drug_metadata


TOOL_DEFINITIONS = [
    {
        "name": "get_drug_metadata",
        "description": (
            "Look up high-level baseline regulatory facts for a specific drug. "
            "Returns drug class, indication, baseline FDA status/pathway, and baseline EMA status/pathway. "
            "Use this first when a query mentions a specific drug to establish baseline regulatory posture."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "drug_id": {
                    "type": "STRING",
                    "description": "Generic name or brand name of the drug (e.g. 'aducanumab', 'Aduhelm', 'lecanemab', 'bevacizumab', 'olaparib').",
                }
            },
            "required": ["drug_id"],
        },
    },
    {
        "name": "search_fda",
        "description": (
            "Search United States FDA drug labels, approval summaries, and accelerated approval guidance. "
            "Returns top matching text chunks with chunk_ids. "
            "Use when answering questions about US regulatory decisions, clinical trial requirements, or FDA labeling."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Semantic search query describing the US regulatory fact, clinical trial, or endpoint.",
                },
                "n_results": {
                    "type": "INTEGER",
                    "description": "Number of relevant chunks to retrieve (default: 3).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_ema",
        "description": (
            "Search European Medicines Agency (EMA) SmPCs, assessment reports, and conditional authorisation guidance. "
            "Returns top matching text chunks with chunk_ids. "
            "Use when answering questions about European Union regulatory decisions, CHMP opinions, or EU population restrictions."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Semantic search query describing the EU regulatory fact, CHMP rationale, or SmPC section.",
                },
                "n_results": {
                    "type": "INTEGER",
                    "description": "Number of relevant chunks to retrieve (default: 3).",
                },
            },
            "required": ["query"],
        },
    },
]


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatches tool calls from the LLM to the matching Python function
    and returns a structured observation dictionary.
    """
    if tool_name == "get_drug_metadata":
        drug_id = arguments.get("drug_id", "")
        result = get_drug_metadata(drug_id)
        if result is None:
            return {"error": f"No metadata found for drug '{drug_id}'."}
        return {"drug_metadata": result}

    elif tool_name == "search_fda":
        query = arguments.get("query", "")
        n_results = int(arguments.get("n_results", 3))
        results = search_fda(query=query, n_results=n_results)
        return {"fda_chunks": results}

    elif tool_name == "search_ema":
        query = arguments.get("query", "")
        n_results = int(arguments.get("n_results", 3))
        results = search_ema(query=query, n_results=n_results)
        return {"ema_chunks": results}

    else:
        return {"error": f"Unknown tool: '{tool_name}'"}
