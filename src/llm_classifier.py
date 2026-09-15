"""
LLM few-shot intent classifier via xAI Grok.

Every call returns (result, LLMResponse). Callers MUST check
response.status before trusting the result. There is no silent fallback
to a default label — a failed call propagates as an error so the
escalation policy can catch it.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
import random
from dotenv import load_dotenv
from openai import OpenAI

from src.project_paths import repo_path

load_dotenv()

# --- Provider config ----------------------------------------------------

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"
CACHE_DIR = repo_path("reports", "llm_cache")

_client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url=GROQ_BASE_URL,
)


# --- Response contract --------------------------------------------------

@dataclass
class LLMResponse:
    status: str            # "success" | "auth_error" | "rate_limited" |
                           # "server_error" | "invalid_output" | "network_error"
    error: str | None = None
    provider: str = "groq"
    model: str = GROQ_MODEL
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "success"


# --- Disk cache (keyed on system + prompt) ------------------------------

def _cache_key(system: str, prompt: str) -> str:
    return hashlib.sha256((system + "\n---\n" + prompt).encode()).hexdigest()

def _cache_get(key: str) -> dict | None:
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def _cache_put(key: str, payload: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, key + ".json")
    with open(path, "w") as f:
        json.dump(payload, f)


# --- Prompt -------------------------------------------------------------

ALLOWED = [
    "delivery_delay_or_missing",
    "wrong_or_damaged_item",
    "return_or_refund",
    "payment_or_gift_card_issue",
    "subscription_or_membership",
    "account_access_or_security",
    "order_status_or_tracking",
    "device_or_app_technical",
    "website_or_app_ux",
    "customer_service_quality",
    "general_complaint",
    "how_to_or_feature_question",
    "availability_or_eligibility_query",
]

DEFS = """- delivery_delay_or_missing: package late, stuck, overdue, never arrived, OR
    handed to the wrong person/left exposed (misdelivery)
- wrong_or_damaged_item: wrong item shipped, arrived broken/defective, or a
    promised digital entitlement (preorder bonus, code) never received
- return_or_refund: wants to return, refund not received, pickup failed
- payment_or_gift_card_issue: charged twice, overcharge, gift-card, card declined
- subscription_or_membership: Prime charge, membership cancellation, renewal confusion
- account_access_or_security: locked out, can't log in, hacked, password, unauthorized access
- order_status_or_tracking: a NEUTRAL status/informational question about an
    EXISTING order, with NO frustration framing -- e.g. "has this shipped
    yet?". If the message expresses frustration or complains the item is
    late, that is delivery_delay_or_missing instead.
- device_or_app_technical: Echo/Kindle/Alexa/Prime Video, app crash, won't sync
- website_or_app_ux: site broken, checkout error, page won't load, new design complaints
- customer_service_quality: agent behaviour, no callback, rude support, phone/chat failures
- general_complaint: frustration with no specific named issue, or a complaint
    that doesn't fit any category above
- how_to_or_feature_question: a question about HOW TO USE a feature or
    product you already have access to -- "how do I do X with something
    I own/can already access", with zero complaint framing
- availability_or_eligibility_query: a NEUTRAL question about WHETHER or
    WHEN something is available or you're eligible for it -- NOT about an
    existing order. Covers: shipping to a location ("do you deliver to
    Turkey?"), stock/restock timing ("when's this back in stock?"),
    content-release timing ("when is season 2 out?"), pre-order/format
    eligibility, promo/trial eligibility ("can I get a free trial?"),
    and product policy/certification questions. The dividing line from
    how_to_or_feature_question: how_to assumes you already have/can access
    the thing and want to use it; this intent is about whether you CAN
    get/access it at all. The dividing line from order_status_or_tracking:
    order_status is about a specific order you've already placed; this is
    about a product/service/promo in general, before or independent of
    placing an order.
"""

FEW_SHOT = """Example 1:
Tweet: "My order was supposed to arrive Tuesday and it's Friday."
Label: delivery_delay_or_missing

Example 2:
Tweet: "How can your delivery person hand my package to 'someone in the area'?!"
Label: delivery_delay_or_missing

Example 3:
Tweet: "Charged twice for the same item, need a refund."
Label: payment_or_gift_card_issue

Example 4:
Tweet: "How do I enable two-factor authentication?"
Label: how_to_or_feature_question

Example 5 (NEGATIVE -- question-shaped but NOT how_to):
Tweet: "Why do you send incorrect tracking updates? #falseservice"
Label: order_status_or_tracking
Reasoning: A question mark alone doesn't make this how_to -- it's a
complaint about tracking accuracy on an existing order.

Example 6:
Tweet: "Do you deliver to Turkey?"
Label: availability_or_eligibility_query

Example 7:
Tweet: "When is season 2 of this show going to be available?"
Label: availability_or_eligibility_query

Example 8 (contrast with Example 7 -- looks similar but IS how_to):
Tweet: "How do I switch between multi-room audio and single-device playback
on my two Echoes?"
Label: how_to_or_feature_question

Example 9 (looks like order_status but is really a delay complaint):
Tweet: "Ordered a case on Sunday and it's still not here."
Label: delivery_delay_or_missing

Example 10:
Tweet: "Your customer service is useless, no one ever calls back."
Label: customer_service_quality

Example 11:
Tweet: "I'm never shopping here again."
Label: general_complaint
"""

SYSTEM = (
    "You are a support-intent classifier. "
    "You must respond with JSON only, in the format requested."
)

PROMPT_TEMPLATE = """Classify the customer tweet into exactly one of these labels:

{defs}

{examples}

Now classify this tweet. Respond with JSON:
{{"intent": "<one of the labels>", "confidence": <0.0-1.0>, "reasoning": "<short>"}}

Tweet: "{text}"
"""


# --- Error mapping ------------------------------------------------------

def _status_from_exception(e: Exception) -> str:
    # openai.APIStatusError carries .status_code
    code = getattr(e, "status_code", None)
    if code == 401 or code == 403:
        return "auth_error"
    if code == 429:
        return "rate_limited"
    if code is not None and 500 <= code < 600:
        return "server_error"
    return "network_error"


# --- Core call ----------------------------------------------------------

def _call_raw(prompt: str, max_tokens: int = 512) -> tuple[dict | None, LLMResponse]:
    key = _cache_key(SYSTEM, prompt)
    cached = _cache_get(key)
    if cached is not None and cached.get("status") == "success":
        return cached["payload"], LLMResponse(status="success", cached=True)

    try:
        resp = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        return None, LLMResponse(status=_status_from_exception(e), error=str(e))

    try:
        payload = json.loads(resp.choices[0].message.content)
    except Exception as e:
        return None, LLMResponse(status="invalid_output", error=str(e))

    _cache_put(key, {"status": "success", "payload": payload})
    return payload, LLMResponse(status="success")

RETRYABLE = {"rate_limited", "server_error", "network_error"}


def _call_raw_with_backoff(prompt, max_tokens=512, max_attempts=3):
    """
    Wrap _call_raw with exponential backoff for transient errors.
    auth_error, bad_request, and invalid_output are not retried.
    """
    last_resp = None
    for attempt in range(max_attempts):
        payload, resp = _call_raw(prompt, max_tokens=max_tokens)
        if resp.ok:
            return payload, resp
        last_resp = resp
        if resp.status not in RETRYABLE:
            return payload, resp
        if attempt == max_attempts - 1:
            break
        delay = (2 ** attempt) + random.uniform(0, 1)   # 1s, 2s, 4s + jitter
        time.sleep(delay)
    return None, last_resp

# --- Public classifier --------------------------------------------------

def classify(text: str) -> tuple[dict | None, LLMResponse]:
    """
    Returns (result, response).

    On success, result is
        {"intent": <one of ALLOWED>, "confidence": float, "reasoning": str}
        and response.status == "success".

    On failure, result is None. Callers must check response.status;
    there is NO fallback label.
    """
    prompt = PROMPT_TEMPLATE.format(defs=DEFS, examples=FEW_SHOT, text=text)

    payload, resp = _call_raw_with_backoff(prompt)
    if resp.ok and payload and payload.get("intent") in ALLOWED:
        return payload, resp

    # One strict retry only if the failure was a bad label / parse issue.
    if resp.status in {"invalid_output", "success"}:
        strict_prompt = (
            prompt
            + "\n\nYour previous answer was invalid or used a label not in the "
              "allowed list. Return only valid JSON with an allowed intent."
        )
        payload2, resp2 = _call_raw_with_backoff(strict_prompt)
        if resp2.ok and payload2 and payload2.get("intent") in ALLOWED:
            return payload2, resp2
        return None, resp2

    return None, resp


# --- Smoke test ---------------------------------------------------------

if __name__ == "__main__":
    tests = [
        "my package was supposed to arrive 3 days ago and it's still not here",
        "why was I charged $9.99 twice this month",
        "how do I return an item",
        "is this available in Canada?",
    ]
    for t in tests:
        result, resp = classify(t)
        print(f"  status={resp.status}  cached={resp.cached}")
        print(f"  {t}")
        print(f"    -> {result}")
        if resp.error:
            print(f"    error: {resp.error}")
        print()