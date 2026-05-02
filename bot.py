"""
magicpin AI Challenge — Vera Bot
=================================
Merchant AI assistant for the magicpin Vera challenge.

Architecture:
- FastAPI HTTP server exposing all 5 required endpoints
- In-memory context store with versioned idempotency
- Groq (llama-3.1-8b-instant) for fast message composition
- Trigger-kind routing, auto-reply detection, intent-handoff
- Multi-turn conversation state management

Requires env var: GROQ_API_KEY
"""

import os
import time
import json
import uuid
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()  # loads .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#print(GROQ_API_KEY)

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vera_bot")

app = FastAPI(title="Vera Bot")
START_TIME = time.time()

# ── in-memory stores ──────────────────────────────────────────────────────────
# contexts[(scope, context_id)] = {version, payload}
contexts: dict[tuple[str, str], dict] = {}
# conversations[conversation_id] = {merchant_id, customer_id, turns: [...], ended}
conversations: dict[str, dict] = {}
# suppression set: suppression_key -> True
sent_suppressions: set[str] = set()
# per-tick dedup: (merchant_id, conversation_id) already actioned this tick
tick_actioned: set[str] = set()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def get_context(scope: str, cid: str) -> Optional[dict]:
    entry = contexts.get((scope, cid))
    return entry["payload"] if entry else None

def context_counts() -> dict:
    counts: dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REPLY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

AUTO_REPLY_PATTERNS = [
    "thank you for contacting",
    "aapki jaankari ke liye bahut-bahut shukriya",
    "main ek automated assistant",
    "i am an automated",
    "this is an automated",
    "aapki madad ke liye shukriya",
    "our team will get back to you",
    "we will contact you shortly",
    "hamari team aapko contact karegi",
    "message received",
]

def is_auto_reply(message: str) -> bool:
    m = message.lower()
    return any(p in m for p in AUTO_REPLY_PATTERNS)

def is_explicit_intent(message: str) -> bool:
    """Detects when merchant signals they want to join/proceed."""
    triggers = [
        "join", "judrna", "jodna", "let's do it", "go ahead", "proceed",
        "ok karein", "haan", "yes", "sign me up", "register", "enroll",
        "main chahta", "main chahti", "interested", "chalega",
    ]
    m = message.lower()
    return any(t in m for t in triggers)

def is_disinterest(message: str) -> bool:
    """Detects hard no / not interested."""
    signals = [
        "not interested", "no thanks", "nahi chahiye", "band karo",
        "stop", "unsubscribe", "mat bhejo", "mujhe nahi chahiye",
        "leave me alone", "go away", "remove", "do not contact",
        "don't contact", "nahi chahiye",
    ]
    m = message.lower()
    return any(s in m for s in signals)


# ─────────────────────────────────────────────────────────────────────────────
# LLM COMPOSITION  (Groq — llama-3.1-8b-instant)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Vera, magicpin's merchant AI assistant. You talk to Indian merchants on WhatsApp.

RULES (follow every single one):
1. Be specific: anchor every message on ONE concrete verifiable fact (number, date, headline, stat).
2. Voice match: clinical-peer for dentists/doctors, energetic for salons/gyms, warm for restaurants. Hindi-English code-mix is encouraged.
3. ONE primary CTA at the end. Binary (YES/STOP) for action triggers; open-ended question for info triggers.
4. Never fabricate data. Only use facts present in the context given to you.
5. No generic "10% off" offers — always use service+price format ("Haircut @ ₹99").
6. No preamble ("I hope you're doing well"). Lead with the hook.
7. No promotional yelling (ALL CAPS, "AMAZING!").
8. Do NOT re-introduce yourself after the first message.
9. Do NOT repeat a message body verbatim that was already sent.
10. Keep it WhatsApp-friendly: concise, scannable, warm.
11. Compulsion levers to USE: specificity, loss aversion, social proof, effort externalization, curiosity, reciprocity, single binary CTA.
12. Anti-patterns to AVOID: multiple CTAs, buried CTA, long preambles, promotional tone for clinical categories, hallucinated data.

OUTPUT FORMAT (JSON only, no markdown):
{
  "body": "the WhatsApp message body",
  "cta": "open_ended" | "binary_yes_stop" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "short:dedup:key",
  "rationale": "one sentence: why this message, what compulsion lever"
}"""


async def call_groq(user_prompt: str, system: str = SYSTEM_PROMPT) -> dict:
    """Call Groq API (llama-3.1-8b-instant) and parse the JSON response."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY env var not set")

    payload = {
        "model": "llama-3.1-8b-instant",
        "max_tokens": 1000,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    }
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=payload,
        )
        r.raise_for_status()
        data = r.json()

    raw = data["choices"][0]["message"]["content"]

    # strip possible ```json fences
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip().rstrip("```").strip()

    return json.loads(clean)





def build_compose_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    conversation_history: Optional[list] = None,
) -> str:
    parts = []

    # ── Category ──
    parts.append("=== CATEGORY CONTEXT ===")
    parts.append(f"Category: {category.get('slug', 'unknown')} ({category.get('display_name', '')})")
    voice = category.get("voice", {})
    parts.append(f"Voice/tone: {voice.get('tone', '')} | code-mix: {voice.get('code_mix', '')}")
    parts.append(f"Taboo words: {', '.join(voice.get('vocab_taboo', []))}")
    catalog = category.get("offer_catalog", [])
    if catalog:
        parts.append(f"Offer catalog: {', '.join(o['title'] for o in catalog[:5])}")
    ps = category.get("peer_stats", {})
    if ps:
        parts.append(f"Peer stats: avg_ctr={ps.get('avg_ctr')}, avg_rating={ps.get('avg_rating')}, avg_reviews={ps.get('avg_review_count')}")
    digest = category.get("digest", [])
    if digest:
        parts.append("Recent digest items:")
        for d in digest[:3]:
            parts.append(f"  - [{d.get('source','')}] {d.get('title','')} (n={d.get('trial_n','?')})")
    seasonal = category.get("seasonal_beats", [])
    if seasonal:
        parts.append(f"Seasonal beats: {'; '.join(s.get('note','') for s in seasonal[:2])}")
    trend = category.get("trend_signals", [])
    if trend:
        for t in trend[:2]:
            parts.append(f"Trend signal: '{t.get('query','')}' {int(t.get('delta_yoy',0)*100)}% YoY")

    # ── Merchant ──
    parts.append("\n=== MERCHANT CONTEXT ===")
    identity = merchant.get("identity", {})
    parts.append(f"Name: {identity.get('name', 'Merchant')}")
    parts.append(f"Location: {identity.get('locality', '')}, {identity.get('city', '')}")
    parts.append(f"Language pref: {identity.get('languages', ['en'])}")
    sub = merchant.get("subscription", {})
    parts.append(f"Subscription: {sub.get('status', 'unknown')}, {sub.get('days_remaining', '?')} days remaining, plan={sub.get('plan', '?')}")
    perf = merchant.get("performance", {})
    if perf:
        parts.append(f"Performance (30d): views={perf.get('views_30d','?')}, calls={perf.get('calls_30d','?')}, ctr={perf.get('ctr','?')}, directions={perf.get('directions_30d','?')}")
        delta = perf.get("deltas_7d", {})
        if delta:
            parts.append(f"7d deltas: {json.dumps(delta)}")
    offers = merchant.get("offers", [])
    if offers:
        active = [o for o in offers if o.get("status") == "active"]
        parts.append(f"Active offers: {', '.join(o.get('title','') for o in active[:3])}")
    signals = merchant.get("signals", [])
    if signals:
        parts.append(f"Derived signals: {', '.join(str(s) for s in signals[:5])}")
    cust_agg = merchant.get("customer_aggregate", {})
    if cust_agg:
        parts.append(f"Customer aggregate: active={cust_agg.get('active_count','?')}, lapsed={cust_agg.get('lapsed_count','?')}, retention_6mo={cust_agg.get('retention_6mo_pct','?')}")

    # Previous Vera conversation
    conv_hist = merchant.get("conversation_history", {})
    last_turns = conv_hist.get("last_turns", []) if isinstance(conv_hist, dict) else []
    if last_turns:
        parts.append("Last Vera conversation turns (most recent last):")
        for t in last_turns[-3:]:
            parts.append(f"  [{t.get('role','?')}]: {str(t.get('text',''))[:120]}")

    # ── Trigger ──
    parts.append("\n=== TRIGGER CONTEXT ===")
    parts.append(f"Trigger ID: {trigger.get('id','?')}")
    parts.append(f"Kind: {trigger.get('kind','?')} | Scope: {trigger.get('scope','?')} | Urgency: {trigger.get('urgency','?')}/5")
    payload = trigger.get("payload", {})
    if payload:
        parts.append(f"Payload: {json.dumps(payload)[:400]}")

    # ── Customer (optional) ──
    if customer:
        parts.append("\n=== CUSTOMER CONTEXT ===")
        cid = customer.get("identity", {})
        parts.append(f"Customer: {cid.get('name','?')}, lang={cid.get('language_pref','en')}")
        rel = customer.get("relationship", {})
        parts.append(f"Relationship: first_visit={rel.get('first_visit','?')}, last_visit={rel.get('last_visit','?')}, visits={rel.get('visits_total','?')}")
        parts.append(f"State: {customer.get('state','?')}")
        prefs = customer.get("preferences", {})
        if prefs:
            parts.append(f"Preferences: {json.dumps(prefs)}")
        parts.append("send_as MUST be 'merchant_on_behalf'")

    # ── In-flight conversation history ──
    if conversation_history:
        parts.append("\n=== CURRENT CONVERSATION SO FAR ===")
        for turn in conversation_history[-6:]:
            parts.append(f"  [{turn['from']}]: {turn['msg'][:200]}")

    # ── Task ──
    parts.append("\n=== YOUR TASK ===")
    trigger_kind = trigger.get("kind", "generic")
    scope = trigger.get("scope", "merchant")

    if customer or scope == "customer":
        parts.append("Compose a customer-facing WhatsApp message FROM the merchant TO their customer.")
        parts.append("Use 'merchant_on_behalf' for send_as.")
    else:
        parts.append("Compose a Vera → Merchant WhatsApp message.")
        parts.append("Use 'vera' for send_as.")

    if trigger_kind in ("research_digest", "category_research_digest_release"):
        parts.append("Frame around the most relevant digest finding for THIS merchant's patient/customer cohort.")
    elif trigger_kind in ("perf_spike",):
        parts.append("Celebrate the spike, anchor on the specific number, then suggest ONE action to sustain it.")
    elif trigger_kind in ("perf_dip",):
        parts.append("Use loss aversion: their number dropped. Offer ONE concrete fix. Don't catastrophize.")
    elif trigger_kind in ("recall_due", "customer_lapsed_soft"):
        parts.append("Recall message: specific service they had last time, time since last visit, 2 slot options. Binary CTA (1/2 for slots or YES).")
    elif trigger_kind in ("dormant_with_vera",):
        parts.append("Re-engagement: use curiosity + one interesting insight from their account. Ask one question.")
    elif trigger_kind in ("milestone_reached",):
        parts.append("Celebrate the milestone (name the number). Suggest ONE next step to capitalize.")
    elif trigger_kind in ("festival_upcoming",):
        parts.append("Festival tie-in: specific offer from catalog + urgency. Keep it warm not hype.")
    elif trigger_kind in ("scheduled_recurring",):
        parts.append("Curiosity-driven ask: pose ONE interesting question about their business this week.")
    elif trigger_kind in ("competitor_opened",):
        parts.append("Competitor awareness: frame as opportunity, not threat. Suggest one differentiator action.")

    parts.append("\nRespond ONLY with the JSON object. No extra text, no markdown fences.")
    return "\n".join(parts)


def build_reply_prompt(
    category: dict,
    merchant: dict,
    conversation: dict,
    incoming_message: str,
    auto_reply_count: int,
) -> str:
    parts = []
    name = merchant.get("identity", {}).get("name", "the merchant")
    lang = merchant.get("identity", {}).get("languages", ["en"])

    parts.append(f"You are Vera replying to {name} (lang: {lang}).")
    parts.append(f"Incoming message: \"{incoming_message}\"")

    history = conversation.get("turns", [])
    if history:
        parts.append("\nConversation so far:")
        for t in history[-5:]:
            parts.append(f"  [{t['from']}]: {t['msg'][:200]}")

    parts.append(f"\nauto_reply_count_so_far: {auto_reply_count}")

    if is_auto_reply(incoming_message):
        parts.append("\nDETECTED: This looks like a WhatsApp Business auto-reply canned response.")
        if auto_reply_count >= 2:
            parts.append("You've already tried twice after auto-replies. Gracefully EXIT the conversation.")
            parts.append('Return: {"action": "end", "body": null, "cta": null, "rationale": "auto-reply detected 3x; graceful exit"}')
        else:
            parts.append("Try ONE more time with a very specific hook that a human would notice.")
            parts.append("Make it hard to ignore if a human sees it.")
    elif is_disinterest(incoming_message):
        parts.append("\nDETECTED: Merchant signaled disinterest. Exit gracefully with a warm close.")
        parts.append('Return action="end" with a brief warm farewell body.')
    elif is_explicit_intent(incoming_message):
        parts.append("\nDETECTED: Merchant signaled YES / intent to proceed. DO NOT ask qualifying questions.")
        parts.append("Switch to ACTION mode immediately. Tell them exactly what you will do next, concretely.")
    else:
        parts.append("\nRespond helpfully and advance the conversation. ONE clear next step.")

    offers = merchant.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    if active_offers:
        parts.append(f"Available active offers to reference: {', '.join(o.get('title','') for o in active_offers[:3])}")

    category_slug = category.get("slug", "")
    voice = category.get("voice", {})
    parts.append(f"Maintain voice: {voice.get('tone','peer')} for {category_slug}.")

    parts.append("\nRespond ONLY with JSON:")
    parts.append('{"action": "send"|"wait"|"end", "body": "...", "cta": "open_ended"|"binary_yes_stop"|"none", "rationale": "..."}')
    parts.append("For action=wait: also add \"wait_seconds\": N")
    parts.append("For action=end: body should be a warm farewell (or null if auto-reply exit).")
    parts.append("No markdown fences.")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER ELIGIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def is_trigger_eligible(trg: dict) -> bool:
    """Check if a trigger should be acted on."""
    # Already suppressed
    sk = trg.get("suppression_key", "")
    if sk and sk in sent_suppressions:
        return False
    # Expired
    expires = trg.get("expires_at")
    if expires:
        try:
            exp = datetime.fromisoformat(expires.replace("Z", "+00:00"))
            if exp < datetime.now(timezone.utc):
                return False
        except Exception:
            pass
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": context_counts(),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Pro",
        "team_members": ["Challenger"],
        "model": "llama-3.1-8b-instant (Groq)",
        "approach": (
            "4-context structured prompt composer with trigger-kind routing, "
            "auto-reply detection, intent-handoff, and multi-turn conversation state."
        ),
        "contact_email": "challenger@example.com",
        "version": "2.0.0",
        "submitted_at": utcnow(),
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Must be one of {valid_scopes}"},
        )
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]},
        )
    contexts[key] = {"version": body.version, "payload": body.payload}
    log.info(f"Stored {body.scope}/{body.context_id} v{body.version}")
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": utcnow(),
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    tick_actioned.clear()
    actions = []

    for trg_id in body.available_triggers:
        if len(actions) >= 20:
            break

        trg = get_context("trigger", trg_id)
        if not trg:
            continue
        if not is_trigger_eligible(trg):
            continue

        merchant_id = trg.get("merchant_id") or trg.get("payload", {}).get("merchant_id")
        if not merchant_id:
            continue

        # dedup: one action per merchant per tick
        if merchant_id in tick_actioned:
            continue

        merchant = get_context("merchant", merchant_id)
        if not merchant:
            continue

        category_slug = merchant.get("category_slug") or merchant.get("identity", {}).get("category_slug")
        if not category_slug:
            # try to derive from merchant id naming convention
            for slug in ["dentists", "salons", "restaurants", "gyms", "pharmacies"]:
                if slug[:3] in merchant_id.lower():
                    category_slug = slug
                    break
        category = get_context("category", category_slug) if category_slug else None
        if not category:
            # find any category as fallback
            for (scope, cid), v in contexts.items():
                if scope == "category":
                    category = v["payload"]
                    break
        if not category:
            continue

        # customer context for customer-scoped triggers
        customer = None
        if trg.get("scope") == "customer":
            customer_id = trg.get("customer_id") or trg.get("payload", {}).get("customer_id")
            if customer_id:
                customer = get_context("customer", customer_id)

        conv_id = f"conv_{merchant_id}_{trg_id}"
        # don't re-start an ended conversation
        conv = conversations.get(conv_id, {})
        if conv.get("ended"):
            continue

        try:
            prompt = build_compose_prompt(category, merchant, trg, customer)
            result = await call_groq(prompt)
        except Exception as e:
            log.error(f"LLM error for {trg_id}: {e}")
            continue

        body_text = result.get("body", "")
        if not body_text:
            continue

        send_as = result.get("send_as", "vera")
        cta = result.get("cta", "open_ended")
        suppression_key = result.get("suppression_key", trg.get("suppression_key", f"{trg_id}"))
        rationale = result.get("rationale", "")

        # mark suppression
        if suppression_key:
            sent_suppressions.add(suppression_key)
        tick_actioned.add(merchant_id)

        # init conversation
        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": trg.get("customer_id"),
            "category_slug": category_slug,
            "turns": [{"from": "vera", "msg": body_text}],
            "auto_reply_count": 0,
            "ended": False,
        }

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": trg.get("customer_id"),
            "send_as": send_as,
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind','generic')}_v1",
            "template_params": [
                merchant.get("identity", {}).get("name", "Merchant"),
                trg.get("kind", "update"),
                body_text[:60],
            ],
            "body": body_text,
            "cta": cta,
            "suppression_key": suppression_key,
            "rationale": rationale,
        })

        log.info(f"Action queued: {conv_id} | {trg.get('kind')} | cta={cta}")

    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply_handler(body: ReplyBody):
    conv = conversations.get(body.conversation_id)
    if not conv:
        # Unknown conversation — start fresh state
        conversations[body.conversation_id] = {
            "merchant_id": body.merchant_id,
            "customer_id": body.customer_id,
            "turns": [],
            "auto_reply_count": 0,
            "ended": False,
        }
        conv = conversations[body.conversation_id]

    if conv.get("ended"):
        return {"action": "end", "rationale": "conversation already ended"}

    # Record incoming
    conv["turns"].append({"from": body.from_role, "msg": body.message})

    # Auto-reply tracking
    if is_auto_reply(body.message):
        conv["auto_reply_count"] = conv.get("auto_reply_count", 0) + 1
    else:
        conv["auto_reply_count"] = 0  # reset on real human reply

    # Hard disinterest → end immediately
    if is_disinterest(body.message):
        conv["ended"] = True
        return {
            "action": "end",
            "body": "Bilkul samajh gaye! Koi baat nahi. Jab bhi zaroorat ho, hum yahaan hain. 🙏",
            "cta": "none",
            "rationale": "Merchant signalled disinterest; graceful exit",
        }

    # Auto-reply 3+ times → exit
    if conv.get("auto_reply_count", 0) >= 3:
        conv["ended"] = True
        return {
            "action": "end",
            "body": None,
            "cta": "none",
            "rationale": "Auto-reply detected 3+ times; exiting to avoid wasting turns",
        }

    # Get contexts for reply composition
    merchant_id = conv.get("merchant_id") or body.merchant_id
    category_slug = conv.get("category_slug")
    merchant = get_context("merchant", merchant_id) if merchant_id else {}
    category = get_context("category", category_slug) if category_slug else {}

    if not merchant:
        merchant = {}
    if not category:
        # fallback: pick any loaded category
        for (scope, cid), v in contexts.items():
            if scope == "category":
                category = v["payload"]
                break
    if not category:
        category = {}

    try:
        prompt = build_reply_prompt(
            category, merchant, conv, body.message, conv.get("auto_reply_count", 0)
        )
        result = await call_groq(prompt)
    except Exception as e:
        log.error(f"LLM reply error for {body.conversation_id}: {e}")
        return {
            "action": "send",
            "body": "Ek second — main check karke aapko bata deti hoon.",
            "cta": "open_ended",
            "rationale": "LLM fallback",
        }

    action = result.get("action", "send")
    reply_body = result.get("body")
    cta = result.get("cta", "open_ended")
    rationale = result.get("rationale", "")

    if action == "end":
        conv["ended"] = True
    elif action == "send" and reply_body:
        conv["turns"].append({"from": "vera", "msg": reply_body})

    response: dict = {"action": action, "rationale": rationale}
    if action == "send":
        response["body"] = reply_body or ""
        response["cta"] = cta
    elif action == "wait":
        response["wait_seconds"] = result.get("wait_seconds", 1800)
    elif action == "end":
        if reply_body:
            response["body"] = reply_body

    return response


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppressions.clear()
    tick_actioned.clear()
    log.info("State wiped on teardown")
    return {"status": "wiped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot:app", host="0.0.0.0", port=8080, reload=True)
