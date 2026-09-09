"""
NODE: Collect lead
PURPOSE: Ask for the details one at a time, in the conversation.
INPUT:   state.lead_type, state.lead (whatever prefill already worked out)
OUTPUT:  {"lead": {...}, "photo_ids": [...]} then on to submit_lead

Three forms come through here — two kinds of estimate and, since a customer was
offered a callback the agent could not actually arrange, a question for the
office. Same loop for all of them; the field list is the only difference.

WHY ONE QUESTION AT A TIME
--------------------------
This used to open a nine-field modal. It worked, and it felt like being handed a
form — the conversation stopped and paperwork started. People abandon paperwork.

So each field is now its own turn: the agent asks, the customer answers, the
agent asks the next thing. Same fields, same validation, same email at the end.

WHY THE ORDER IS FIXED AND PYTHON DECIDES IT
--------------------------------------------
The obvious alternative is to let a model run the interview — decide what to ask
next, parse whatever comes back. It interviews better and it is the wrong choice
here, for two reasons. It can loop or drift, so there is no guarantee the form
ever finishes. And whatever it decides a field contains ends up in an email a
manager acts on, with nothing between the model and the customer.

This loop always terminates, always asks for exactly what the office needs, and
costs nothing per question. The one model call in this flow is prefill, which
runs before this node and only fills fields the customer can see are wrong.

RE-ENTRANCY
-----------
Every interrupt() below re-runs from the top of this node on each resume, so
there is deliberately nothing here that touches the outside world — no model
call, no write, no upload. Just replaying answers already given.
"""

import logging
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END
from langgraph.types import Command, interrupt

from agent.state import SupportState
from schemas import lead_form
from services import config

logger = logging.getLogger(__name__)


CONFIRM_ASK = (
    "That's everything. Have a quick look before I send it over — a manager "
    "works straight from this, so it's worth catching anything that's off."
)

#: The read-back on a question is shorter because the form was. Three answers do
#: not need to be introduced as though they were nine.
CONFIRM_QUESTION = (
    "Here's what I'm sending over — worth a quick look, since this is what "
    "they'll be answering."
)

CANCEL_REPLY = (
    "No problem, I'll leave it there. If you change your mind just say so — or "
    f"the office is on {config.COMPANY_PHONE} any day from 6am."
)

RESTART_OPENING = "No problem — let's go through it again from the top."

#: Added to the opening when prefill already worked some answers out. No leading
#: space: the join adds it. A string whose meaning depends on surrounding
#: whitespace is one a translator will hand back trimmed.
QUICK_SUFFIX = "I've got a few bits already, so this'll be quick."

#: They pressed skip on something we cannot send the lead without.
NEEDED = "I do need this one, sorry — what should I put down?"

#: Said before repeating the question, when they answered it with a question.
DEFLECT = (
    "Good question — let me get these details over first and a manager will "
    "cover that properly."
)

#: The buttons and placeholders around the question. Not prose, but the customer
#: reads them, and a Russian question with an English "Skip" under it is the seam
#: showing. Sent to the browser already translated, so the page itself holds no
#: copy that could drift out of sync with what was translated.
UI_LINES = {
    "skip": "Skip",
    "not_sure": "Not sure yet",
    "send": "Send it over",
    "restart": "Start over",
    "add_photos": "Add photos",
    "send_photos": "Send these",
    "skip_photos": "Skip for now",
    "type_answer": "Type your answer…",
    "or_type": "Or type something…",
    "or_change": "Or tell me what to change…",
    # On a card of choices: the message box is still there, and this says so.
    "or_reply": "Or reply directly…",
    "something_else": "Something else",
}

#: The roadmap line above the question. Numbers rather than a bare bar, because
#: "how many more of these are there" is the actual question someone has three
#: questions into an interview, and a bar only answers it approximately.
STEP_LABEL = "Question {step} of {total}"

#: Counted rather than pluralised. "1 photo / 2 photos" needs a plural rule per
#: language — Russian alone has three forms — and getting it wrong looks worse
#: than a colon.
PHOTOS_ATTACHED = "Photos attached: {n}"

#: Everything in THIS file that a customer reads, for agent/nodes/translate.py.
#:
#: The interview's other copy lives on the fields themselves and is enumerated by
#: `lead_form.customer_strings`. These few lines belong to the loop rather than to
#: any one field, so they are listed here and the translate node concatenates the
#: two. A line that is in neither list is a line that only ever comes out in
#: English.
INTERVIEW_LINES = (
    CONFIRM_ASK,
    CONFIRM_QUESTION,
    CANCEL_REPLY,
    RESTART_OPENING,
    QUICK_SUFFIX,
    NEEDED,
    DEFLECT,
    STEP_LABEL,
    PHOTOS_ATTACHED,
    *UI_LINES.values(),
)


def _confirm_ask(lead_type: str) -> str:
    return CONFIRM_QUESTION if lead_type == "question" else CONFIRM_ASK


def _chrome(tr, step: int, total: int) -> dict:
    """The buttons and the roadmap line, in the customer's language."""
    return {
        "ui": {key: tr(text) for key, text in UI_LINES.items()},
        # Filled here rather than in the browser: the translated template can put
        # the numbers wherever that language puts them.
        "step_label": tr(STEP_LABEL).format(step=step, total=total),
    }


def _translator(state: SupportState):
    """
    English string in, the customer's language out.

    Keyed by the English text itself, with the text as its own fallback, so a
    phrase the translator missed comes out in English rather than blank. An
    English conversation gets the identity function and never notices this
    exists.
    """
    book = state.get("phrasebook") or {}
    if not book:
        return lambda text: text
    return lambda text: book.get(text, text)

#: An answer that is plainly a question rather than an answer. Someone mid-form
#: who suddenly asks "wait, how much is 3 movers?" should not end up with that
#: sentence recorded as their zip code.
#:
#: Never on a textarea. Those are the boxes that ASK for something open — "what
#: would you like me to have them answer?", "anything else I should pass on?" —
#: and deflecting a question typed into one is deflecting the answer we asked
#: for. It is also the whole content of a question lead.
def _looks_like_a_question(text: str, field: lead_form.Field_) -> bool:
    if field.kind in ("select", "date", "textarea"):
        return False
    stripped = str(text).strip()
    return stripped.endswith("?") and len(stripped.split()) > 2


def _ask(payload: dict) -> dict:
    """
    One turn of the interview. Returns the resume payload from the browser.

    Shape coming back is one of:
      {"answer": "..."}   they typed something
      {"skip": true}      they skipped an optional question
      {"photo_ids": [..]} the photo step
      {"cancelled": true} they backed out
    """
    reply = interrupt(payload)
    return reply if isinstance(reply, dict) else {"answer": str(reply or "")}


def collect_lead(state: SupportState) -> Command[Literal["submit_lead", "__end__"]]:
    lead_type = state.get("lead_type") or "estimate"
    spec = lead_form.spec(lead_type)
    wants_photos = lead_form.wants_photos(lead_type)
    prefilled = dict(state.get("lead") or {})

    # Built by agent/nodes/translate.py, one node earlier. Reading it is not a
    # side effect, so it is safe in here where a model call would not be: this
    # node re-runs from the top on every answer.
    tr = _translator(state)

    restarted = False

    # One turn of this loop is one full pass through the questions. "Start over"
    # at the confirmation just goes round again, which works because each pass
    # issues fresh interrupt() calls and LangGraph feeds the new answers into
    # them in order.
    while True:
        # A restart asks for everything, including whatever prefill had worked
        # out from the conversation. Someone who chose to start over means it —
        # silently keeping three answers they never typed is not starting over.
        answers: dict[str, str] = {} if restarted else dict(prefilled)

        # `first` gates the opening line, and a restart needs its own opening —
        # without resetting it the customer clicks "start over" and is dropped
        # straight back onto "First off, what's your name?" with no acknowledgement
        # that anything happened.
        first = True
        opening = tr(spec["opening"])
        if restarted:
            opening = tr(RESTART_OPENING)
        elif answers.get("name") and spec["opening_again"]:
            # Their contact details are already in hand from an earlier lead in
            # this same conversation, so the opening must not offer to take them.
            opening = tr(spec["opening_again"])
        elif answers and lead_type != "question":
            # Not on a question: the only thing prefilled there is the question
            # itself, and "I've got a few bits already" about the sentence they
            # just typed reads as though we mean their contact details.
            opening += " " + tr(QUICK_SUFFIX)

        # Recomputed on every pass rather than once, because an either/or answer
        # decides what is still to come: someone who asks to be phoned is never
        # asked for an email address.
        total = lead_form.steps_remaining(lead_type, answers) + int(wants_photos) + 1
        step = 0

        for field in lead_form.fields_for(lead_type):
            if answers.get(field.name):
                continue
            # Checked HERE, not when the list was built. `contact_method` is
            # answered part-way down this loop, and it is what decides whether
            # the phone or the email question is the next one asked.
            if not field.applies(answers):
                continue

            step += 1
            question = field.question(answers, tr)
            # Fires at most once per field. Someone whose actual surname trips
            # the check should not be stuck in a loop being asked to clarify it.
            clarified = False
            while True:
                reply = _ask(
                    {
                        "type": "question",
                        "lead_type": lead_type,
                        "field": field.localized(tr),
                        "message": question,
                        # Only ever on the very first question, so the opening
                        # line is not repeated every time validation fails.
                        "opening": opening if first else "",
                        "skippable": not field.required,
                        "step": step,
                        "total": total,
                        **_chrome(tr, step, total),
                    }
                )
                first = False

                if reply.get("cancelled"):
                    logger.info("Lead form: cancelled at %s", field.name)
                    return Command(
                        update={"messages": [AIMessage(content=tr(CANCEL_REPLY))]}, goto=END
                    )

                if reply.get("skip"):
                    if field.required:
                        question = tr(NEEDED)
                        continue
                    break

                value = str(reply.get("answer") or "").strip()

                # "not sure yet" is a real answer to the date question.
                if field.kind == "date" and lead_form.is_undecided(value):
                    value = ""

                # And "no" is a real answer to "anything else?". Recording it
                # sends a manager a line that looks like information and isn't.
                if field.kind == "textarea" and not field.required and lead_form.is_nothing(value):
                    value = ""

                # The options are buttons, but the message box never goes away.
                # Someone who types "email" at a question offering "Email" has
                # answered it.
                value = lead_form.coerce_option(field, value)

                # "what" is a perfectly valid string and would pass every check
                # we have — straight into an email as somebody's name. It is
                # never an answer, and always a sign the question was badly put,
                # so say it plainer rather than record it.
                if not clarified and lead_form.sounds_confused(value):
                    clarified = True
                    question = field.rephrased(answers, tr)
                    logger.info("Lead form: rephrasing %s", field.name)
                    continue

                if _looks_like_a_question(value, field):
                    question = f"{tr(DEFLECT)} {field.question(answers, tr)}"
                    continue

                if message := lead_form.validate_one(field, value):
                    question = tr(message)
                    continue

                if value:
                    answers[field.name] = value
                break

        # Photos, second to last. They need the customer to go and do something,
        # and by this point they have already invested eight answers — asking
        # first loses everyone who hasn't got photos to hand.
        #
        # Skipped entirely on a question, which has no move to photograph. A step
        # whose only sensible answer is "skip" is a step that should not be there.
        photo_ids: list[str] = []
        if wants_photos:
            step += 1
            question = tr(lead_form.PHOTO_STEP["ask"])
            # Asked at most twice. The second ask says WHY, because the reason is
            # genuinely persuasive and nobody has heard it yet — but a customer
            # who skips again has answered the question, and an agent that keeps
            # pushing past that is the reason people close the tab.
            for attempt in range(2):
                reply = _ask(
                    {
                        "type": "photos",
                        "lead_type": lead_type,
                        "message": question,
                        "opening": opening if first else "",
                        "skippable": True,
                        "step": step,
                        "total": total,
                        **_chrome(tr, step, total),
                    }
                )
                first = False
                if reply.get("cancelled"):
                    logger.info("Lead form: cancelled at photos")
                    return Command(
                        update={"messages": [AIMessage(content=tr(CANCEL_REPLY))]}, goto=END
                    )
                photo_ids = [str(p)[:32] for p in (reply.get("photo_ids") or [])][
                    : config.MAX_UPLOADS_PER_THREAD
                ]
                if photo_ids or attempt:
                    break
                logger.info("Lead form: photos skipped, asking once more")
                question = tr(lead_form.PHOTO_STEP["insist"])

        # The read-back. Nothing has left the building yet, and this is the last
        # moment it can be corrected — after this a person acts on it, and a
        # digit wrong in the phone number is a job nobody can call back about.
        step += 1
        cleaned = lead_form.clean(lead_type, answers)
        reply = _ask(
            {
                "type": "confirm",
                "lead_type": lead_type,
                "message": tr(_confirm_ask(lead_type)),
                "summary": lead_form.summary_for(lead_type, cleaned, tr),
                "photo_count": len(photo_ids),
                "photos_line": tr(PHOTOS_ATTACHED).format(n=len(photo_ids)),
                "step": step,
                "total": total,
                **_chrome(tr, step, total),
            }
        )

        if reply.get("cancelled"):
            logger.info("Lead form: cancelled at the read-back")
            return Command(update={"messages": [AIMessage(content=tr(CANCEL_REPLY))]}, goto=END)

        if reply.get("restart"):
            logger.info("Lead form: customer restarted at the read-back")
            restarted = True
            continue

        break

    logger.info("Lead form: confirmed (%s, %d photos)", lead_type, len(photo_ids))
    return Command(
        update={
            "lead_type": lead_type,
            "lead": cleaned,
            "photo_ids": photo_ids,
        },
        goto="submit_lead",
    )
