import gradio as gr
from witness.witness_rzero import WitnessRZero
import app_math as app_math

# Instantiate WitnessRZero – change device to "cuda" if GPU is available
wrz = WitnessRZero(device="cpu")

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    # Build conversation history in OpenAI-style message format
    messages = [{"role": "system", "content": system_message}]
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})
    messages.append({"role": "user", "content": message})

    # Concatenate all into a single prompt for WitnessRZero
    prompt = ""
    for m in messages:
        prompt += f"{m['role'].capitalize()}: {m['content']}\n"

    response = ""
    # Stream the output from WitnessRZero.generate()
    for token in wrz.client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        part = token  # huggingface_hub’s stream yields token text chunks
        response += part
        yield response

# Build the Gradio ChatInterface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

if __name__ == "__main__":
    demo.launch()



