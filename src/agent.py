"""
Agent Core: ReAct Loop Module
Implements the hand-written while loop that drives dynamic tool selection,
trajectory logging, draft report generation, and verification retries.
"""

import os
import time
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
from google.genai.models import Models

# Set SDK internal guard flag so it knows manual function calling is intentional
Models._logged_afc_warning = True

from src.config import GEMINI_API_KEY, GEMINI_AGENT_MODEL, call_gemini_with_retry
from src.tools import TOOL_DEFINITIONS, execute_tool
from src.verify import verify_draft_report, VerificationReport

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

    def __init__(self, model_name: str = GEMINI_AGENT_MODEL):
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

            # 1. Call Gemini LLM with current message history and available tools, with 4s timeoff in case of quota exceeded
            response = call_gemini_with_retry(
                client = self.client,
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=gemini_tools,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
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

    def run(self, query: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Full autonomous pipeline:
        1. Runs ReAct loop to retrieve facts and draft report.
        2. Runs independent Verifier pass on cited claims.
        3. If verification fails, feeds feedback back to agent for self-correction (max 2 retries).
        """
        react_result = self.run_react_loop(query)
        draft_report = react_result["draft_report"]
        retrieved_chunks = react_result["retrieved_chunks"]
        trajectory = react_result["trajectory"]

        verification_history: List[Dict[str, Any]] = []
        final_report = draft_report
        retry_count = 0

        # Step 2: Verification and Self-Correction Loop
        while retry_count <= max_retries:
            # Run independent verification pass
            verdict: VerificationReport = verify_draft_report(
                draft_report=final_report,
                retrieved_chunks=retrieved_chunks,
                model_name=self.model_name,
            )

            verification_history.append({
                "attempt": retry_count + 1,
                "all_supported": verdict.all_supported,
                "audited_claims": [c.model_dump() for c in verdict.audited_claims],
                "feedback": verdict.feedback_for_correction,
            })

            # If all claims are supported by source chunks, we are done
            if verdict.all_supported:
                break

            # If unsupported claims exist and we have retries left, trigger correction
            if retry_count < max_retries:
                retry_count += 1
                correction_prompt = f"""You are regenerating a regulatory report following a failed verification audit.

ORIGINAL USER QUERY:
{query}

PREVIOUS DRAFT REPORT:
{final_report}

AUDITOR VERIFICATION FEEDBACK:
{verdict.feedback_for_correction}

INSTRUCTIONS:
Regenerate the full regulatory report adhering strictly to the SYSTEM_PROMPT format:
- **Regulatory Posture:** Must be exactly `HARMONIZED`, `DIVERGENT`, or `INSUFFICIENT EVIDENCE`.
- **Executive Summary:** High-level summary of the findings.
- **FDA Position & Evidence:** Approved indication, pathway, key trial data, and citations [fda_chunk_id].
- **EMA Position & Evidence:** Approved indication or refusal grounds, key trial data, restrictions, and citations [ema_chunk_id].
- **Comparative Synthesis & Rationale:** Cross-jurisdictional rationale.

Fix the specific ungrounded claims identified by the auditor, or explicitly mark them as [INSUFFICIENT EVIDENCE].
"""
                # Request a corrected draft from Gemini
                correction_response = call_gemini_with_retry(
                    client = self.client,
                    model=self.model_name,
                    contents=correction_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                    ),
                )
                final_report = correction_response.text or final_report
            else:
                # Retries exhausted: break and return the audited report
                break

        return {
            "query": query,
            "final_report": final_report,
            "is_verified": verification_history[-1]["all_supported"] if verification_history else False,
            "verification_history": verification_history,
            "trajectory": trajectory,
            "retrieved_chunks": retrieved_chunks,
            "retries_used": retry_count,
        }

