import os

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.groq import Groq

from hybrid_model.models import RaportModel

PROMPT_TEMPLATE = PromptTemplate(
"""You are a spam analyst.
Given the following data, generate a detailed and formal report explaining whether the message is spam or not, and why:

Subject: {subject}
Metadata: {metadata}
Prediction from model: {prediction_model} (confidence {confidence_model:.2f})
Prediction from RAG: {prediction_rag} (confidence {confidence_rag:.2f})

Be formal, clear, and concise.
"""
)

def generate_rapport_body(rapport: RaportModel) -> str:
    prompt = PROMPT_TEMPLATE.format(
        subject=rapport.subject,
        metadata=rapport.metadata,
        prediction_model=rapport.prediction_model,
        confidence_model=rapport.confidence_model,
        prediction_rag=rapport.prediction_rag,
        confidence_rag=rapport.confidence_rag,
    )

    api_key = os.getenv("GROQ_API_KEY")
    assert api_key is not None, "GROQ_API_KEY environment variable not set."

    response = Groq(model="gemma2-9b-it", temperature=0, api_key=api_key).complete(prompt)
    return response.text.strip()