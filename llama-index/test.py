import os

from llama_index.llms.groq import Groq
from dotenv import load_dotenv

load_dotenv()

llm = Groq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"), temperature=0)
response = llm.complete("when second world war ended")

print(response)

