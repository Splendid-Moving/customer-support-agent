"""
How Sam sounds and listens in voice mode — the speaking counterpart to persona.py.

Everything a customer HEARS that is not an approved reply or an interview
question lives here: the tone the speech model is asked for, the few fixed lines
voice mode needs (greeting, hints), and the context the transcriber is given so
it spells our name and local places right. Change the voice here, nowhere else.
"""

from schemas.persona import AGENT_NAME
from services import config

#: How the speech model should deliver every line. Delivery only — the WORDS are
#: always the approved text, never something this prompt could add to.
TTS_INSTRUCTIONS = (
    "Warm, relaxed and friendly, like someone who works in the office of a "
    "family-run Los Angeles moving company and is happy to help. Natural, "
    "conversational pace; not salesy, not overly cheerful. Read prices, phone "
    "numbers and zip codes clearly."
)

#: Context for the transcriber. It biases spelling, not meaning.
STT_PROMPT = (
    f"A customer talking to {AGENT_NAME} from {config.COMPANY_NAME}, a moving "
    "company in Los Angeles, about a home or office move: names, phone numbers, "
    "email addresses, zip codes, dates, apartment sizes."
)

#: Words the transcriber should prefer when it hears something close. Each must
#: be one line with no angle brackets — OpenAI rejects the whole session otherwise.
STT_KEYWORDS = [
    config.COMPANY_NAME,
    AGENT_NAME,
    "studio",
    "bedroom",
    "zip code",
    "Gmail",
]


# ── Fixed lines ──────────────────────────────────────────────────────────────
# Signed by the server and handed to the page once (see /api/voice/config), so
# the browser can say them without being able to say anything else.

GREETING = (
    f"Hi, I'm {AGENT_NAME} from {config.COMPANY_NAME}. What can I help you with?"
)

#: Voice switched on part-way through a conversation. No second hello.
RESUME = "I'm listening — go ahead."

#: The transcriber returned something we could not act on, or a turn failed.
NOT_CAUGHT = "Sorry, I didn't catch that — could you say it again?"

#: At the read-back. A lead a manager acts on is never sent on a maybe, so
#: anything that is not a clear yes or "start over" gets this instead.
CONFIRM_HINT = (
    "If that all looks right, just say \"send it\". Or say \"start over\" to "
    "change something."
)

#: At the photo step. Photos have to be tapped — there is no saying a photo.
PHOTOS_HINT = (
    "You can tap \"Add photos\" to send some from your phone, or just say "
    "\"skip\" and we'll go with what we've got."
)

LINES = {
    "greeting": GREETING,
    "resume": RESUME,
    "not_caught": NOT_CAUGHT,
    "confirm_hint": CONFIRM_HINT,
    "photos_hint": PHOTOS_HINT,
}

#: Which hint is spoken after which kind of interview step. These are English,
#: so they are only added when the interview itself is in English — a Russian
#: read-back followed by an English instruction is worse than no instruction.
HINT_AFTER = {
    "confirm": CONFIRM_HINT,
    "photos": PHOTOS_HINT,
}

#: How we tell the interview is running in English: its button labels are the
#: untranslated originals. (collect_lead sends them already translated.)
ENGLISH_UI = {"send": "Send it over", "skip_photos": "Skip for now", "skip": "Skip"}
