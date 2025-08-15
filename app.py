import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading

import app_math as app_math  # keeping your existing import

# ---- Model setup ----
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

# Automatically map model across available devices (GPU/CPU)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",          # << key change
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True,
    token=HF_TOKEN,
)

# Ensure pad token is set
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def respond(message, history: list[tuple[str, str]], system_message, max_tokens, temperature, top_p):
    # Build chat messages
    messages = [{"role": "system", "content": system_message}]
    for u, a in history:
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    # Tokenize with Zephyr's chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    # Stream generation
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        "inputs": inputs,
        "max_new_tokens": int(max_tokens),
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    partial = ""
    for new_text in streamer:
        partial += new_text
        yield partial


# ---- Gradio UI ----
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
)

if __name__ == "__main__":
    demo.launch()


