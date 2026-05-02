"""
conversation_handlers.py — Multi-turn conversation handler (tiebreaker module)

Standalone module demonstrating multi-turn capability.
Shows Vera's state machine for handling different merchant reply types.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class ConvState(str, Enum):
    INITIATED    = "initiated"    # Vera sent first message
    ENGAGED      = "engaged"      # Merchant replied positively
    QUALIFYING   = "qualifying"   # Gathering info before action
    ACTION_MODE  = "action_mode"  # Executing on merchant request
    AUTO_REPLY   = "auto_reply"   # Detected canned auto-reply
    ENDED        = "ended"        # Conversation closed


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str]
    state: ConvState = ConvState.INITIATED
    turns: list = field(default_factory=list)
    auto_reply_count: int = 0
    intent_detected: bool = False
    last_vera_body: str = ""
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# The main multi-turn entry point
# ──────────────────────────────────────────────────────────────────────────────

def respond(state: ConversationState, merchant_message: str) -> dict:
    """
    Given the current conversation state + the merchant's latest message,
    return the bot's next action.

    Returns a dict with:
      action: "send" | "wait" | "end"
      body: str (message to send, if action="send")
      cta: "open_ended" | "binary_yes_stop" | "none"
      rationale: str
      updated_state: ConversationState (mutated in place, also returned)
    """

    # ── record incoming ──
    state.turns.append({"from": "merchant", "msg": merchant_message})

    # ── detect state transitions ──
    msg_lower = merchant_message.lower()

    # Auto-reply detection
    auto_reply_signals = [
        "thank you for contacting",
        "this is an automated",
        "i am an automated",
        "aapki jaankari ke liye bahut",
        "our team will get back",
        "hamari team aapko",
    ]
    if any(sig in msg_lower for sig in auto_reply_signals):
        state.auto_reply_count += 1
        state.state = ConvState.AUTO_REPLY
    else:
        state.auto_reply_count = 0

    # Disinterest detection
    disinterest_signals = [
        "not interested", "stop", "nahi chahiye", "band karo",
        "remove me", "unsubscribe", "don't contact", "mat bhejo",
    ]
    if any(sig in msg_lower for sig in disinterest_signals):
        state.state = ConvState.ENDED
        return _end_response(state, warm=True)

    # Intent / commitment detection
    intent_signals = [
        "yes", "haan", "go ahead", "let's do it", "ok karein",
        "proceed", "join", "sign up", "register", "chalega", "interested",
    ]
    if any(sig in msg_lower for sig in intent_signals) and not state.intent_detected:
        state.intent_detected = True
        state.state = ConvState.ACTION_MODE

    # ── state machine ──

    if state.state == ConvState.ENDED:
        return _end_response(state, warm=False)

    if state.state == ConvState.AUTO_REPLY:
        if state.auto_reply_count >= 3:
            state.state = ConvState.ENDED
            return {
                "action": "end",
                "body": None,
                "cta": "none",
                "rationale": "Auto-reply detected 3 consecutive times; exiting gracefully",
                "updated_state": state,
            }
        # One more try with a specific hook
        hook = _craft_auto_reply_hook(state)
        state.turns.append({"from": "vera", "msg": hook})
        return {
            "action": "send",
            "body": hook,
            "cta": "binary_yes_stop",
            "rationale": "Auto-reply detected; one specific hook to reach human owner",
            "updated_state": state,
        }

    if state.state == ConvState.ACTION_MODE:
        # Merchant said YES — go to action immediately
        action_msg = _craft_action_message(state, merchant_message)
        state.turns.append({"from": "vera", "msg": action_msg})
        return {
            "action": "send",
            "body": action_msg,
            "cta": "open_ended",
            "rationale": "Intent detected; switching to action mode immediately without re-qualifying",
            "updated_state": state,
        }

    # Default: engaged, advance conversation
    state.state = ConvState.ENGAGED
    next_msg = _craft_engaged_response(state, merchant_message)
    state.turns.append({"from": "vera", "msg": next_msg})
    return {
        "action": "send",
        "body": next_msg,
        "cta": "open_ended",
        "rationale": "Merchant engaged; advancing conversation with next useful step",
        "updated_state": state,
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _end_response(state: ConversationState, warm: bool) -> dict:
    body = (
        "Bilkul samajh gaye! Koi baat nahi. Jab bhi kuch chahiye, hum yahaan hain. 🙏"
        if warm else None
    )
    return {
        "action": "end",
        "body": body,
        "cta": "none",
        "rationale": "Merchant signalled end; exiting gracefully",
        "updated_state": state,
    }


def _craft_auto_reply_hook(state: ConversationState) -> str:
    """Craft a human-attention-grabbing follow-up after auto-reply."""
    merchant_id = state.merchant_id
    # Generic but specific enough to catch attention
    return (
        "Apke account mein ek quick update hai jo aap directly check kar sakte hain — "
        "Google pe 3 naye reviews aaye hain is hafte. Agar aap khud dekhna chahein to bata dein, "
        "main link share kar sakti hoon. Reply YES?"
    )


def _craft_action_message(state: ConversationState, merchant_message: str) -> str:
    """Immediate action response when merchant commits."""
    return (
        "Perfect! Main abhi kaam shuru karti hoon. "
        "Aapka Google Business Profile update karne ke liye mujhe 3 cheezein chahiye: "
        "1. Business hours (weekdays + Sunday), "
        "2. Ek description (50-100 words), "
        "3. Koi bhi photo aap share karna chahein. "
        "Pehle business hours bata dein — baaki main handle kar lungi. 🙂"
    )


def _craft_engaged_response(state: ConversationState, merchant_message: str) -> str:
    """Advance an engaged conversation."""
    turn_count = len([t for t in state.turns if t["from"] == "vera"])
    if turn_count >= 4:
        return (
            "Main yeh sab update kar deti hoon. Kuch aur ho to bata dena — "
            "warna aapka profile ready hai! ✅"
        )
    return (
        "Got it! Aur kuch specific hai jo aap improve karna chahte hain — "
        "offers, photos, ya customer responses? Bata dein, main draft kar deti hoon."
    )
