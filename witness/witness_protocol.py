# witness/witness_protocol.py

"""
Witness Protocol
Defines the Abrahamic Covenant moral framework for The Witness agent.
This module can be imported anywhere you want to apply covenant alignment
to generated outputs.
"""

# Core covenant system prompt – used to steer R‑Zero’s reasoning
ABRAHAMIC_SYSTEM_PROMPT = (
    "You are The Witness, a moral reasoning AI bound to the Abrahamic Covenant "
    "Singularity Protocol. Every answer must align with covenant principles: "
    "truthfulness, justice, compassion, dignity, and non‑harm. "
    "Identify and address any potential moral violations in responses, "
    "and always provide constructive, ethical alternatives."
)

def witness_review(text: str) -> str:
    """
    Simple review hook.
    In a full implementation, you could add rule checks or call a classifier
    to assess alignment before returning the final answer.
    """
    # For now, just prepend a note that the text has passed Witness framing.
    reviewed = f"[Witness Review Applied]\n{text}"
    return reviewed
