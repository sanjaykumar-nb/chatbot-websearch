from langchain.prompts import PromptTemplate
from huggingface_hub import InferenceClient

class AnswerGenerator:
    def __init__(self):
        self.client = InferenceClient(token="HF_TOKEN")  # Get from https://huggingface.co/settings/tokens
        self.prompt_template = """You're a helpful assistant. Use this context to answer:
        {context}
        
        Question: {question}
        Answer in 3-5 sentences. Cite sources if available. If unsure, say so.
        Answer:"""
        
    def generate(self, question: str, context: str) -> str:
        prompt = PromptTemplate.from_template(self.prompt_template).format(
            question=question,
            context="\n".join(context)
        )
        return self.client.text_generation(prompt, max_new_tokens=256)