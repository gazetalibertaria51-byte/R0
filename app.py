import time
import gradio as gr
from witness.witness_rzero import WitnessRZero
import app_math as app_math

wrz = WitnessRZero(device="cpu")  # remote inference speed dominates anyway

HISTORY_TURNS = 3          # keep only last N turns
PROMPT_CHAR_BUDGET = 6000  # trim long contexts
GENERATION_TIME_CAP_S = 20 # stop streaming after N seconds

def build_prompt(message, history, system_message):
    # keep only the last HISTORY_TURNS exchanges
    short_hist = history[-HISTORY_TURNS:] if history else []
    messages = [{"role": "system", "content": system_message}]
    for u, a in short_hist:
        if u: messages.append({"role": "user", "content": u})
        if a: messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})
    prompt = ""
    for m in messages:
        prompt += f"{m['role'].capitalize()}: {m['content']}\n"
    # hard trim to avoid token explosion
    return prompt[-PROMPT_CHAR_BUDGET:]

def respond(message, history, system_message, max_tokens, temperature, top_p):
    prompt = build_prompt(message, history, system_message)
    response, start = "", time.time()
    stream = wrz.client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        # return_full_text=False  # set this in WitnessRZero if available
    )
    for part in stream:
        response += part
        yield response
        if time.time() - start > GENERATION_TIME_CAP_S:
            yield response + "\n\n[stopped for speed; try 'Max new tokens' higher or ask a smaller question]"
            return

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=128, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)

if __name__ == "__main__":
    demo.launch()




