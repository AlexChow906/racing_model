"""
AI agents for the racing pipeline.

- Race analysis agent: a true tool-using agent. Given a value bet, it decides
  which DuckDB tools to call, observes the results, loops, and returns a verdict
  (CONFIRM / NEUTRAL / FADE) with reasoning. It can disagree with the model.
- Commentary: single LLM calls for the morning preview and post-race summary.

Provider-agnostic via the OpenAI-compatible chat completions API. Default Groq.
Swap with LLM_PROVIDER=groq|anthropic|openai in the environment.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.agent_tools import TOOL_SCHEMAS, dispatch, DB_PATH

VALID_TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        # agent needs strong tool use; commentary just writes 2-3 sentences
        "agent_model": "openai/gpt-oss-120b",
        "commentary_model": "openai/gpt-oss-20b",
    },
    "openai": {
        "base_url": None,  # default OpenAI endpoint
        "api_key_env": "OPENAI_API_KEY",
        "agent_model": "gpt-4o",
        "commentary_model": "gpt-4o-mini",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/",
        "api_key_env": "ANTHROPIC_API_KEY",
        "agent_model": "claude-sonnet-4-6",
        "commentary_model": "claude-sonnet-4-6",
    },
}

MAX_AGENT_ITERATIONS = 6


def _get_client():
    """Return (OpenAI client, provider config) for the configured LLM_PROVIDER."""
    from openai import OpenAI

    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    cfg = PROVIDERS.get(provider)
    if cfg is None:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing {cfg['api_key_env']} for provider {provider}")
    kwargs = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs), cfg


def _model_for(role: str, cfg: dict) -> str:
    """Resolve the model for a role ('agent' or 'commentary').

    Uses the role-specific env (AGENT_MODEL / COMMENTARY_MODEL) if set,
    otherwise the provider default for that role.
    """
    role_env = "AGENT_MODEL" if role == "agent" else "COMMENTARY_MODEL"
    return os.environ.get(role_env) or cfg[f"{role}_model"]


def _retry_after_seconds(exc: Exception, default: float = 2.0) -> float:
    """Seconds to wait after a 429, from the Retry-After header or the error text."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            header = resp.headers.get("retry-after")
            if header:
                return float(header) + 0.5
        except (TypeError, ValueError):
            pass
    match = re.search(r"try again in ([\d.]+)\s*s", str(exc))
    if match:
        return float(match.group(1)) + 0.5
    return default


def _chat_with_retry(client, max_retries: int = 6, **kwargs):
    """chat.completions.create with backoff on 429s.

    Groq's free tier caps tokens-per-minute low; a 429 there just means "wait a
    second or two", so honour the retry-after hint and retry rather than crash the
    publish run. Other errors (e.g. the gpt-oss tool_use_failed 400) propagate
    unchanged so their existing handlers still run.
    """
    import openai

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            if attempt == max_retries:
                raise
            time.sleep(min(_retry_after_seconds(exc), 30.0))


# ── Race analysis agent ──────────────────────────────────────────────

ANALYST_SYSTEM = (
    "You are a sharp, sceptical horse racing analyst. A quantitative model has "
    "flagged a horse as a value bet (its win probability is higher than the market "
    "implies). Your job is to investigate using the tools provided and decide whether "
    "the value is real.\n\n"
    "Call tools to check the horse's recent form, going and course records, and the "
    "trainer/jockey form. Look for red flags (poor record on today's ground, weak "
    "recent form, high non-completion rate for jumpers) and green flags (strong "
    "course/going record, in-form yard, improving profile). A run flagged "
    "result_missing is a real start whose result our data does not have. Include it "
    "when counting recent starts — if there are 5 starts and only the latest has no "
    "result, say 'last 5 starts' (not 4) and report the wins and placings you can "
    "see among them, without asserting anything about that missing run. Never invent "
    "a result or tell the reader one is 'pending' or 'not in yet'.\n\n"
    "When you have enough information, STOP calling tools and reply with your "
    "verdict as plain assistant text (do NOT wrap it in a tool/function call). "
    "The reply MUST be valid JSON only, no other text, in this exact shape:\n"
    '{"verdict": "CONFIRM|NEUTRAL|FADE", "confidence": "high|medium|low", '
    '"note": "2-3 sentence punter-facing note, published to subscribers", '
    '"concerns": "frank internal note of any weaknesses or red flags"}\n\n'
    "The model also tells you which signals it rated the horse on. Weigh whether "
    "the evidence you gather actually backs those signals up — if the data confirms "
    "them, lean CONFIRM; if the data contradicts them, lean FADE; if it is mixed, "
    "lean NEUTRAL.\n\n"
    "CONFIRM = evidence supports the bet. NEUTRAL = nothing decisive either way. "
    "FADE = evidence contradicts the bet.\n\n"
    "The `note` and `concerns` fields have different audiences — keep them strictly "
    "separate:\n"
    "- `note` is PUBLISHED to subscribers whenever the verdict is CONFIRM or NEUTRAL. "
    "Write it as a supportive, punter-facing note that gives useful context for the "
    "bet. It must NEVER criticise, second-guess, hedge, or cast doubt on the selection "
    "or the model — no 'however', 'but', 'uncertain', 'unproven', 'risky', not even "
    "for NEUTRAL. For a NEUTRAL pick keep it factual and understated, never negative; "
    "if there is little to say positively, keep it brief rather than padding it with "
    "doubts.\n"
    "- `concerns` is INTERNAL. Put ALL of your scepticism here — weak form, unproven "
    "going, a tough field, anything that gave you pause. It is shown only in a private "
    "review channel for FADE picks, never to subscribers, so be frank.\n\n"
    "Do not use jargon like 'model probability' or 'feature'."
)


def run_race_analysis_agent(bet: dict[str, Any], db_path: str = DB_PATH,
                            verbose: bool = False) -> dict[str, Any]:
    """
    Run the tool-using agent on a single value bet.

    `bet` must include: horse, course, race_time, category, model_prob, back_odds,
    edge, and the IDs horse_id, trainer_id, jockey_id, course_id, race_id, going_code.

    Returns {verdict, confidence, analysis_text, tools_used}.
    """
    client, cfg = _get_client()
    model = _model_for("agent", cfg)

    implied = 1.0 / bet["back_odds"] if bet.get("back_odds") else 0.0
    ids = {
        "horse_id": bet.get("horse_id"),
        "trainer_id": bet.get("trainer_id"),
        "jockey_id": bet.get("jockey_id"),
        "course_id": bet.get("course_id"),
        "race_id": bet.get("race_id"),
        "going_code": bet.get("going_code"),
    }
    raw_signals = bet.get("model_signals")
    signals = raw_signals.strip() if isinstance(raw_signals, str) else ""
    signals_line = f"- Model's strongest signals: {signals}\n" if signals else ""
    user_prompt = (
        f"Value bet to investigate:\n"
        f"- Horse: {bet.get('horse')}\n"
        f"- Course: {bet.get('course')} ({bet.get('race_time')})\n"
        f"- Type: {bet.get('category')}\n"
        f"- Model win chance: {bet.get('model_prob', 0):.0%}\n"
        f"- Market price: {bet.get('back_odds')} (implies {implied:.0%})\n"
        f"- Edge: +{bet.get('edge', 0):.0%}\n"
        f"{signals_line}\n"
        f"IDs to pass to tools (use these exact values):\n"
        f"{json.dumps({k: v for k, v in ids.items() if v is not None}, indent=2)}\n\n"
        f"Investigate (especially whether the model's signals hold up) and return "
        f"your verdict as JSON."
    )

    messages = [
        {"role": "system", "content": ANALYST_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    tools_used: list[str] = []

    for _ in range(MAX_AGENT_ITERATIONS):
        try:
            resp = _chat_with_retry(
                client,
                model=model, messages=messages, tools=TOOL_SCHEMAS,
                tool_choice="auto", temperature=0.3,
            )
        except Exception as exc:
            # gpt-oss on Groq sometimes returns its final answer by "calling" a
            # tool named json, which fails server-side validation. The verdict it
            # generated is in error.failed_generation — recover it.
            recovered = _failed_generation(exc)
            if recovered:
                return _parse_verdict(recovered, tools_used)
            raise
        msg = resp.choices[0].message

        # No tool calls -> this is the final answer
        if not msg.tool_calls:
            return _parse_verdict(msg.content, tools_used)

        # If the model "called" an unknown tool (e.g. json) to emit its answer,
        # treat its arguments as the verdict instead of dispatching.
        real_calls = [tc for tc in msg.tool_calls if tc.function.name in VALID_TOOL_NAMES]
        if not real_calls:
            for tc in msg.tool_calls:
                if "verdict" in (tc.function.arguments or ""):
                    return _parse_verdict(tc.function.arguments, tools_used)
            return _parse_verdict(msg.content, tools_used)

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in real_calls
            ],
        })

        for tc in real_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = dispatch(name, args, db_path)
            tools_used.append(name)
            if verbose:
                print(f"    tool: {name}({args}) -> {str(result)[:120]}", flush=True)
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps(result, default=str)[:4000],
            })

    # Hit iteration cap — force a final answer without tools
    messages.append({
        "role": "user",
        "content": "Stop investigating. Give your final verdict as JSON now, "
                   "as plain text (not a tool call).",
    })
    try:
        final = _chat_with_retry(
            client,
            model=model, messages=messages, temperature=0.3,
        )
    except Exception as exc:
        recovered = _failed_generation(exc)
        if recovered:
            return _parse_verdict(recovered, tools_used)
        raise
    return _parse_verdict(final.choices[0].message.content, tools_used)


def _failed_generation(exc: Exception) -> str | None:
    """Pull the model's attempted output from a Groq tool_use_failed 400 error."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("failed_generation"):
            return err["failed_generation"]
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            err = resp.json().get("error", {})
            if err.get("failed_generation"):
                return err["failed_generation"]
        except Exception:
            pass
    return None


def _parse_verdict(content: str | None, tools_used: list[str]) -> dict[str, Any]:
    text = (content or "").strip()
    # Strip code fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"verdict": "NEUTRAL", "confidence": "low",
                "analysis_text": text[:300] or "No analysis produced.",
                "tools_used": tools_used}
    # If wrapped as a tool call envelope {"name":..., "arguments":{...}}, unwrap it
    if "verdict" not in data and isinstance(data.get("arguments"), dict):
        data = data["arguments"]
    verdict = str(data.get("verdict", "NEUTRAL")).upper()
    if verdict not in ("CONFIRM", "NEUTRAL", "FADE"):
        verdict = "NEUTRAL"
    note = str(data.get("note", "")).strip()
    concerns = str(data.get("concerns", "")).strip()
    legacy = str(data.get("analysis", "")).strip()  # older single-field shape
    # FADE goes to the private review channel, so surface the frank concerns there.
    # CONFIRM/NEUTRAL are published to subscribers — only ever the clean note.
    analysis_text = (concerns or legacy or note) if verdict == "FADE" else (note or legacy)
    return {
        "verdict": verdict,
        "confidence": str(data.get("confidence", "medium")).lower(),
        "analysis_text": analysis_text or "No analysis produced.",
        "tools_used": tools_used,
    }


# ── Commentary (single calls) ────────────────────────────────────────

def generate_morning_preview(posted_bets: list[dict[str, Any]], pnl_context: str) -> str:
    """One LLM call. Writes a short morning preview from the posted picks + P&L context."""
    client, cfg = _get_client()
    model = _model_for("commentary", cfg)

    lines = []
    for b in posted_bets:
        lines.append(
            f"- {b.get('horse')} ({b.get('course')} {b.get('race_time')}, {b.get('category')}): "
            f"back {b.get('back_odds')}, edge +{b.get('edge', 0):.0%}, "
            f"verdict {b.get('verdict', 'NEUTRAL')}"
        )
    bet_block = "\n".join(lines) if lines else "No qualifying picks today."

    prompt = (
        "Write a concise morning racing preview (3-4 sentences) for a value betting "
        "Discord. Lead with the headline pick (highest edge). Mention how many bets "
        "across which categories. Reference the running P&L if relevant. Confident, "
        "not arrogant. No betting jargon.\n\n"
        f"Today's posted picks:\n{bet_block}\n\n"
        f"P&L context: {pnl_context}\n"
    )
    resp = _chat_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You write punchy, honest racing previews."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )
    return resp.choices[0].message.content.strip()


def generate_post_race_summary(results: list[dict[str, Any]], pnl_context: str) -> str:
    """One LLM call. Writes a short post-race summary from settled results + P&L."""
    client, cfg = _get_client()
    model = _model_for("commentary", cfg)

    lines = []
    for r in results:
        outcome = "WON" if r.get("won") else "LOST"
        back = r.get("back")
        price = f"backed {back}" if back else f"SP {r.get('sp')}"
        lines.append(
            f"- {r.get('horse')} ({r.get('course')} {r.get('time')}): "
            f"{price} — {outcome} {r.get('profit', 0):+.2f}u"
        )
    result_block = "\n".join(lines) if lines else "No settled bets."

    # Judge the day by its OWN net result, computed here, so the model can't mistake
    # the running drawdown for the day's outcome — a profitable day is a good day.
    n = len(results)
    winners = sum(1 for r in results if r.get("won"))
    staked = sum(float(r.get("stake", 1.0)) for r in results)
    profit = sum(float(r.get("profit", 0.0)) for r in results)
    roi = (profit / staked * 100) if staked else 0.0
    verdict = "a winning day" if profit > 0 else "a losing day" if profit < 0 else "a breakeven day"
    day_line = (f"The day's net result: {n} bet(s), {winners} winner(s), "
                f"P&L {profit:+.2f}u, ROI {roi:+.1f}% — {verdict}.")

    prompt = (
        "Write a concise post-race summary (2-3 sentences) for a value betting "
        "Discord. Judge whether it was a good or bad day ONLY by the day's net P&L "
        "below: positive = a good/winning day, negative = a bad/losing day. Do NOT "
        "call a profitable day bad because of the running totals. Lead with that "
        "verdict, mention notable winners with the price taken, and be honest on "
        "losing days. No jargon.\n\n"
        f"{day_line}\n\n"
        f"The day's results:\n{result_block}\n\n"
        f"Running P&L (background only, NOT the day's result): {pnl_context}\n"
    )
    resp = _chat_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You write honest racing result summaries."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )
    return resp.choices[0].message.content.strip()


def generate_reviewed_pick_note(bet: dict[str, Any]) -> str:
    """One LLM call. Writes a neutral, punter-facing note for a pick the analyst
    flagged but a human reviewed and approved for publication.

    The original FADE reasoning is deliberately NOT passed in: an approved pick is
    published to subscribers, so the note must present it cleanly without recycling
    the analyst's criticism of the model.
    """
    client, cfg = _get_client()
    model = _model_for("commentary", cfg)

    raw_signals = bet.get("model_signals")
    signals = raw_signals.strip() if isinstance(raw_signals, str) else ""
    signals_line = f"- Model's strongest signals: {signals}\n" if signals else ""

    prompt = (
        "Write a short (2-3 sentence) punter-facing note giving useful context for "
        "this value pick. Keep it factual and understated — do NOT criticise, "
        "second-guess, or cast doubt on the selection or the model, and don't oversell "
        "it either. Write only about the horse and the bet: do NOT mention an analyst, "
        "any review or approval, or that the pick was flagged. No jargon like 'model "
        "probability' or 'feature'.\n\n"
        f"- Horse: {bet.get('horse')}\n"
        f"- Course: {bet.get('course')} ({bet.get('race_time') or bet.get('time')})\n"
        f"- Type: {bet.get('category')}\n"
        f"- Model win chance: {bet.get('model_prob', 0):.0%}\n"
        f"- Price: {bet.get('back_odds')} | Edge +{bet.get('edge', 0):.0%}\n"
        f"{signals_line}"
    )
    resp = _chat_with_retry(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You write concise, neutral racing pick notes."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    return resp.choices[0].message.content.strip()
