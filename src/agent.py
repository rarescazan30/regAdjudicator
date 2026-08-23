"""
Agent Core: ReAct Loop Module
Implements the hand-written while loop that drives dynamic tool selection,
trajectory logging, draft report generation, and verification retries.
"""

import os
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.tools import TOOL_DEFINITIONS, execute_tool


# System Prompt: Strict Guardrails and Output Structure
SYSTEM_PROMPT = """You are the Regulatory Adjudicator Agent, an expert in comparing US (FDA) and European Union (EMA) drug approvals, indications, and clinical trial regulations.

CRITICAL OPERATING RULES:
1. NEVER answer from model memory. Every factual assertion must be backed by data retrieved from tools.
2. DUAL-JURISDICTION REQUIREMENT: When evaluating a drug, you MUST check BOTH FDA and EMA sources before reaching a conclusion, unless the user query is explicitly restricted to one jurisdiction (e.g. 'In the US only...').
3. DYNAMIC TOOL USAGE: Call 'get_drug_metadata' first to establish baseline status, then use 'search_fda' and 'search_ema' to retrieve specific clinical trial evidence and safety warnings.
4. MANDATORY CITATIONS: Every factual statement in your report MUST cite the specific chunk ID it came from, formatted as [chunk_id] (e.g. [aducanumab_fda_chunk_01]).
5. STRUCTURE YOUR REPORT EXACTLY AS FOLLOWS:
   - **Regulatory Posture:** Must be exactly one of: `HARMONIZED`, `DIVERGENT`, or `INSUFFICIENT EVIDENCE`.
   - **Executive Summary:** High-level summary of the cross-jurisdictional findings.
   - **FDA Position & Evidence:** Approved indication, pathway, key trial data, and citations [fda_chunk_id].
   - **EMA Position & Evidence:** Approved indication or refusal grounds, key trial data, restrictions, and citations [ema_chunk_id].
   - **Comparative Synthesis & Rationale:** Why the two regulatory agencies converged or diverged (e.g. surrogate endpoints vs overall survival, risk tolerance for ARIA).
"""


class RegulatoryAgent:
    """
    Hand-crafted ReAct agent
    Manages conversational turn buffer, dynamic tool calls, and trace logs.
    """

    def __init__(self, model_name: str = GEMINI_MODEL):
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add it to your .env file."
            )
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = model_name

    def run_react_loop(
        self, query: str, max_steps: int = 6
    ) -> Dict[str, Any]:
        """
        Executes the ReAct loop for a single query.
        Returns:
            - draft_report (str)
            - trajectory (List[str] - names of tools called in order)
            - tool_trace (List[Dict] - tool names, arguments, and return observations)
            - retrieved_chunks (Dict[str, str] - mapping of chunk_id to chunk_text for verification)
        """
        trajectory: List[str] = []
        tool_trace: List[Dict[str, Any]] = []
        retrieved_chunks: Dict[str, str] = {}

        # Format tool definitions for Google GenAI SDK
        gemini_tools = [{"function_declarations": TOOL_DEFINITIONS}]

        # Initialize message history with system instruction and user query
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)],
            )
        ]

        step = 0
        draft_report = ""

        while step < max_steps:
            step += 1

            # 1. Call Gemini LLM with current message history and available tools
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=gemini_tools,
                    temperature=0.1,  # Low temperature for deterministic factual extraction
                ),
            )

            # Check if model requested one or more tool calls
            function_calls = response.function_calls

            if not function_calls:
                # The model decided it has enough facts and emitted the final report text!
                draft_report = response.text or ""
                break

            # Append the model's tool-call intent to conversation history
            contents.append(response.candidates[0].content) # model *can* return multiple candidates, but we only use the first one

            function_responses = []
            for call in function_calls:
                tool_name = call.name
                tool_args = dict(call.args) if call.args else {}

                trajectory.append(tool_name)

                # Execute Python function in src.tools
                observation = execute_tool(tool_name, tool_args)

                tool_trace.append({
                    "step": step,
                    "tool": tool_name,
                    "args": tool_args,
                    "observation": observation,
                })

                # Cache retrieved chunk text so the Verifier can audit them later
                if "fda_chunks" in observation:
                    for chunk in observation["fda_chunks"]:
                        retrieved_chunks[chunk["chunk_id"]] = chunk["text"]
                if "ema_chunks" in observation:
                    for chunk in observation["ema_chunks"]:
                        retrieved_chunks[chunk["chunk_id"]] = chunk["text"]

                # Format observation as a FunctionResponse part for the LLM
                function_responses.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": observation},
                    )
                )

            # Append tool outputs back to message history so LLM sees the observation
            contents.append(
                types.Content(
                    role="user",
                    parts=function_responses,
                )
            )

        return {
            "query": query,
            "draft_report": draft_report,
            "trajectory": trajectory,
            "tool_trace": tool_trace,
            "retrieved_chunks": retrieved_chunks,
        }


if __name__ == "__main__":
    agent = RegulatoryAgent()
    query = "Is Aducanumab (Aduhelm) authorized for prescription in both the US and EU markets? Explain why they diverged."
    
    print(f"User Query: {query}\n")
    print("Executing ReAct Loop...")
    result = agent.run_react_loop(query)
    
    print("\n--- Tool Call Trajectory ---")
    print(" -> ".join(result["trajectory"]))
    
    print("\n--- Draft Report ---")
    print(result["draft_report"])
