import torch
import gradio as gr
from transformers import pipeline

# Load a summarization pipeline with pretrained model
pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

# Define a Gradio interface
def summarize_text(text):
    summary = pipe(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Create Gradio app
app = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, label="Enter text to summarize"),
    outputs=gr.Textbox(label="Summary"),
    title="AI Text Summarizer",
    description="Summarizes long text using a pretrained Hugging Face model."
)

# Launch app
app.launch(share=True)
