from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS
import gradio as gr
import os
import re

# Initialize AI Client
client = InferenceClient(token=os.environ.get("HF_TOKEN"))

def web_search(query: str, max_results=3):
    """Fetch web results using DuckDuckGo"""
    with DDGS() as ddgs:
        return [
            {"title": r["title"], "content": r["body"], "link": r["href"]}
            for r in ddgs.text(query, max_results=max_results)
        ]

def format_response(text):
    """Format AI response for readability and structure"""

    # Remove duplicate words
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)

    # Bold important keywords
    important_words = [
        "AI", "machine learning", "deep learning", "algorithm", "training", 
        "neural network", "accuracy", "data", "model", "features"
    ]
    for word in important_words:
        text = re.sub(rf"\b{word}\b", f"<strong>{word}</strong>", text, flags=re.IGNORECASE)

    # Convert lists into numbered points
    text = re.sub(r"\d+\.\s(.+)", r"<br><br>\1", text)  # Ensure spacing
    text = re.sub(r"•\s(.+)", r"<br><br>\1", text)  # Handle bullet points if present

    # Add a line space between subheadings and content
    text = re.sub(r"(.*?):\s", r"<br><strong>\1</strong><br>", text)  

    # Reduce unnecessary spaces & line breaks
    text = re.sub(r"\n\s*\n", "<br><br>", text)  

    return text.strip()

def generate_response(query):
    """Fetch web results and generate AI response"""
    results = web_search(query)

    # Format search context
    context = " ".join([f"<strong>{res['title']}</strong>: {res['content']}" for res in results])

    # AI-generated answer
    prompt = f"""Provide a complete and structured answer. Use numbered points and clear subheadings.
    
    Query: {query}
    Context: {context}
    
    Answer:"""

    raw_response = client.text_generation(
        prompt,
        max_new_tokens=800,  # Increased length for completeness
        temperature=0.5  
    ).strip()

    # Validate response
    if not raw_response or len(raw_response) < 20:
        raw_response = "⚠️ AI could not generate a complete response. Please try rephrasing your query."

    return format_response(raw_response), results

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("**#💬 HI! I'm ZYLO...**")

    with gr.Row():
        with gr.Column(scale=3):  
            chatbot = gr.Chatbot(height=450, bubble_full_width=False, show_copy_button=True)
            msg = gr.Textbox(label="Type a message...", interactive=True, autofocus=True)
            clear = gr.Button("🗑️ Clear Chat")
        with gr.Column(scale=1):  
            link_section = gr.HTML("<h4>🔗 Relevant Links</h4><p>(No links yet.)</p>", elem_id="link-box")

    chat_history = gr.State([])

    def respond(message, chat):
        response, sources = generate_response(message)

        # Format sources as HTML (clickable links)
        formatted_links = "<h4>🔗 Relevant Links</h4><ul>"
        for res in sources:
            formatted_links += f'<li><a href="{res["link"]}" target="_blank">{res["title"]}</a></li>'
        formatted_links += "</ul>" if sources else "<p>(No links available.)</p>"

        # Append chat messages
        chat.append(("You", message))  
        chat.append(("🤖 ZYLO", response))  

        return "", chat, formatted_links

    msg.submit(respond, [msg, chat_history], [msg, chatbot, link_section])
    clear.click(lambda: ("", [], "<h4>🔗 Relevant Links</h4><p>(No links yet.)</p>"), None, [msg, chatbot, link_section], queue=False)

app.launch(share=True)
