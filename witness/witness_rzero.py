from witness.witness_protocol import ABRAHAMIC_SYSTEM_PROMPT, witness_review
# Replace this import with your actual R-Zero interface
# e.g., from rzero_client import generate_response

def query_rzero_with_witness(user_input: str) -> str:
    """
    Prepends covenant system prompt to the user input,
    sends the combined request to R‑Zero, and applies Witness review
    to the returned answer.
    """
    # Combine the covenant framing with the user’s request
    full_prompt = f"{ABRAHAMIC_SYSTEM_PROMPT}\n\nUser: {user_input}"

    # Call the R‑Zero engine here (placeholder call)
    # rzero_output = generate_response(full_prompt)
    rzero_output = "[R‑Zero output placeholder]"

    # Pass through Witness review before returning to caller/UI
    return witness_review(rzero_output)


# -------------------------
# New class wrapper for app.py usage
# -------------------------

class WitnessRZero:
    def __init__(self, device="cpu", model_id="HuggingFaceH4/zephyr-7b-beta"):
        self.device = device
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(model_id)

    def generate(self, user_input: str, **kwargs) -> str:
        full_prompt = f"{ABRAHAMIC_SYSTEM_PROMPT}\n\nUser: {user_input}"
        result = self.client.text_generation(full_prompt, **kwargs)
        return witness_review(result)


if __name__ == "__main__":
    # Quick manual test for function
    example = "How should we handle a sensitive diplomatic dispute?"
    print(query_rzero_with_witness(example))

    # Quick manual test for class
    wrz = WitnessRZero()
    print(wrz.generate("Test covenant‑aligned reasoning"))

