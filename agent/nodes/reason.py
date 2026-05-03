from __future__ import annotations

import json
import logging

import httpx
from pydantic import BaseModel, Field, TypeAdapter

from agent.state import AgentState, DriftChangeType, DriftEvent
from config import settings
from db.queries import get_section_text_for_filing

logger = logging.getLogger(__name__)


class LlmReasonResponse(BaseModel):
    is_significant: bool
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    change_type: DriftChangeType


def _format_prompt(
    *,
    ticker: str,
    section_key: str,
    previous_period: str | None,
    latest_period: str | None,
    cosine_score: float,
    old_text: str,
    new_text: str,
) -> str:
    return (
        "You are analyzing SEC filing section semantic drift.\n"
        "\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{\n'
        '  "is_significant": boolean,\n'
        '  "confidence": number,  // 0..1\n'
        '  "summary": string,      // what changed\n'
        '  "change_type": string   // one of: '
        '"content_added" | "content_removed" | "content_reframed" | '
        '"new_risk_or_uncertainty" | "other"\n'
        "}\n"
        "\n"
        f"Ticker: {ticker}\n"
        f"Section: {section_key}\n"
        f"Previous period: {previous_period or 'unknown'}\n"
        f"Latest period: {latest_period or 'unknown'}\n"
        f"Cosine similarity (embeddings): {cosine_score:.6f}\n"
        "\n"
        "Old section text:\n"
        f"{old_text}\n"
        "\n"
        "New section text:\n"
        f"{new_text}\n"
        "\n"
        "Guidance:\n"
        "- If changes are primarily cosmetic/rephrasing, set is_significant=false.\n"
        "- If there are substantive meaning/commitment/risk changes, set is_significant=true.\n"
        "- Provide a short, concrete summary of the drift.\n"
        "- Choose change_type that best fits.\n"
    )


async def _call_openai(prompt: str) -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # OpenAI: choices[0].message.content contains JSON string if response_format used.
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, dict):
        return content

    raise RuntimeError("Unexpected OpenAI response format")


async def _call_anthropic(prompt: str) -> dict:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": settings.ANTHROPIC_API_KEY or "",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": settings.LLM_MODEL,
        "max_tokens": 600,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    content_blocks = data.get("content") or []
    if not content_blocks:
        raise RuntimeError("Unexpected Anthropic response (no content blocks).")

    text = content_blocks[0].get("text")
    if not isinstance(text, str):
        raise RuntimeError("Unexpected Anthropic response (content text missing).")

    return json.loads(text)


async def _call_llm(prompt: str) -> LlmReasonResponse:
    if settings.OPENAI_API_KEY:
        parsed = await _call_openai(prompt)
    elif settings.ANTHROPIC_API_KEY:
        parsed = await _call_anthropic(prompt)
    else:
        raise RuntimeError(
            "No LLM API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )

    adapter = TypeAdapter(LlmReasonResponse)
    return adapter.validate_python(parsed)


async def reason_node(state: AgentState) -> AgentState:
    diffs = state.get("diffs", [])
    sections = state.get("sections", {})
    _ = state.get("embeddings", {})  # kept for contract compatibility

    previous_period = state.get("previous_period_of_report")
    latest_period = state.get("latest_period_of_report")

    filing_id_old = state.get("filing_id_old")
    previous_sections_text = (
        get_section_text_for_filing(int(filing_id_old)) if filing_id_old is not None else {}
    )

    state["drift_events"] = []

    for diff in diffs:
        section_key = diff["section_key"]
        cosine_score = diff["cosine_score"]

        # Skip LLM entirely if cosine similarity is very high (cosmetic/trivial).
        if cosine_score > settings.TRIVIAL_THRESHOLD:
            continue

        new_text = sections.get(section_key, "").strip()
        if not new_text:
            continue

        old_text = previous_sections_text.get(section_key, "").strip()

        prompt = _format_prompt(
            ticker=state["ticker"],
            section_key=section_key,
            previous_period=previous_period,
            latest_period=latest_period,
            cosine_score=cosine_score,
            old_text=old_text,
            new_text=new_text,
        )

        try:
            llm_resp = await _call_llm(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "reason_node LLM call failed for %s section=%s: %s",
                state["ticker"],
                section_key,
                exc,
            )
            state["errors"].append(f"reason_node: {exc!r}")
            continue

        # Store the LLM verdict for persistence.
        event: DriftEvent = {
            "section_key": section_key,
            "filing_id_old": state.get("filing_id_old"),
            "filing_id_new": state["filing_id_new"],
            "cosine_score": cosine_score,
            "is_significant": bool(llm_resp.is_significant),
            "confidence": float(llm_resp.confidence),
            "summary": llm_resp.summary,
            "change_type": llm_resp.change_type,
        }
        state["drift_events"].append(event)

    return state
