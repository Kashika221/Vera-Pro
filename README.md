# Vera Pro — Submission README

## Approach

I built this around a structured 4-context prompt architecture — every message is composed from the full category, merchant, trigger, and (optionally) customer context fed into an LLM. Nothing is templated generically; each message is generated fresh from the actual data for that merchant at that moment.

I used **Groq (llama-3.1-8b-instant)** for inference because of its low latency, which keeps every response well inside the 30s timeout even under load.

### Key design decisions

**Trigger-kind routing**: I wrote specialized composition instructions per trigger type rather than one generic prompt. Research digests get framed as peer-collegial knowledge shares; performance dips use loss aversion framing; recall messages anchor on exact last-visit dates and available slot times. This avoids the "one prompt fits all" problem that makes most bots sound generic.

**Anti-auto-reply detection**: I studied the production Vera examples in the brief and noticed the biggest wasted turns were on WhatsApp Business canned replies. My `/v1/reply` handler pattern-matches known auto-reply phrases in both English and Hindi, tracks consecutive hits, and exits after the second one rather than burning more turns.

**Intent-handoff**: One of the explicit anti-patterns in the brief was re-qualifying after a merchant already said yes. I added explicit intent detection — when a merchant says "yes", "let's do it", "chalega", or any commitment phrase, the bot immediately switches to action mode and tells them the concrete next step. No more qualifying questions after commitment.

**Specificity enforcement**: I baked the anti-patterns directly into the system prompt as hard rules — no "10% off" framing, always service+price format ("Haircut @ ₹99"), peer stat citations required, source attribution for research items. The LLM is also explicitly told to never fabricate numbers not present in the context.

**Language matching**: I default to Hindi-English code-mix for merchants whose `languages` field includes `hi`, and pure English otherwise. This was one of the patterns I noticed in the gold-standard examples in the brief.

### What I'd improve with more time

1. **Retrieval over digest items** — I'd embed all digest items per category and retrieve the top-3 most relevant to each merchant's specific patient/customer cohort before composing. Right now I pass all digest items; retrieval would make the anchor even more precise.
2. **Persistent conversation memory** — move conversation history to Redis so the bot never repeats a message even if it restarts between test phases.
3. **Self-scoring before send** — generate 2 candidate messages per trigger, score them internally against the 5 rubric dimensions, return only the higher scorer.

### What additional context would have helped most

Real-time Google Business Profile completeness data per merchant — the actual profile score, which photo categories are missing, and recent review velocity. Every message I write tries to anchor on something the merchant can immediately verify; live GBP data would make that anchor even stronger and more personal.

## Tech stack

- Python 3.11 + FastAPI + uvicorn
- Groq API — llama-3.1-8b-instant (temperature=0)
- In-memory context store with versioned idempotency
- httpx for async API calls

## Running

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Test locally:
```bash
export BOT_URL=http://localhost:8080
python judge_simulator.py
```
