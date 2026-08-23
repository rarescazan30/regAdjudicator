"""
Verification Module
Independent second-pass LLM evaluator that parses citations in draft reports,
cross-references each claim against the raw retrieved chunk text,
and returns structured pass/fail verdicts with actionable feedback for retries.
"""

import os
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_VERIFIER_MODEL, call_gemini_with_retry

class ClaimAudit(BaseModel):
    claim_text: str = Field(
        description="The factual sentence or statement made in the draft report."
    )
    cited_chunk_id: str = Field(
        description="The chunk_id cited by the draft for this claim."
    )
    is_supported: bool = Field(
        description="True if the provided chunk text directly supports the claim; False if ungrounded, exaggerated, or contradicted."
    )
    rationale: str = Field(
        description="Concise explanation citing specific evidence from the chunk."
    )


class VerificationReport(BaseModel):
    all_supported: bool = Field(
        description="True if all audited claims are supported; False if any claim failed."
    )
    audited_claims: List[ClaimAudit] = Field(
        description="List of claim verification audits."
    )
    feedback_for_correction: str = Field(
        description="If all_supported is False, provide clear, actionable instructions detailing what claims to fix or drop. If True, leave empty."
    )


VERIFIER_SYSTEM_PROMPT = """You are a strict Regulatory Verification Auditor.
Your job is to audit a draft report against the actual retrieved source chunks to eliminate hallucinations.

RULES:
1. Extract every factual claim in the draft report that has a [chunk_id] citation.
2. For each cited claim, compare it strictly against the provided chunk text for that chunk_id.
3. Mark 'is_supported' as TRUE ONLY IF the chunk explicitly states or directly proves the fact.
4. Mark 'is_supported' as FALSE IF:
   - The chunk does not mention the fact.
   - The numbers, percentages, trial names, or dates differ from the chunk.
   - The claim exaggerates or misinterprets the regulatory status.
5. If any claim fails, generate detailed 'feedback_for_correction' explaining the exact mismatch so the draft can be corrected.
"""


def verify_draft_report(
    draft_report: str,
    retrieved_chunks: Dict[str, str],
    model_name: str = GEMINI_VERIFIER_MODEL,
) -> VerificationReport:
    """
    Independent verification pass.
    Takes the draft report and the dictionary of cached chunk texts,
    and returns a structured Pydantic VerificationReport.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Format the ground truth chunks into a readable ledger
    chunks_reference_text = "\n\n".join(
        [f"--- CHUNK ID: {cid} ---\n{ctext}" for cid, ctext in retrieved_chunks.items()]
    )
    # prompt containing source of truth, draft report and verification instructions
    prompt = f"""
=== RETRIEVED SOURCE CHUNKS (GROUND TRUTH) ===
{chunks_reference_text}

=== DRAFT REPORT TO AUDIT ===
{draft_report}

Perform the factual audit of all cited claims in the draft report against the ground truth chunks.
"""

    response = call_gemini_with_retry(
        client=client,
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=VERIFIER_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=VerificationReport,
            temperature=0.0,  # 0.0 temperature for deterministic evaluation
        ),
    )

    # Parse the verified JSON string into our typed Pydantic object
    return VerificationReport.model_validate_json(response.text)
