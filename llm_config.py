import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---- LLM setup (Gemini through the OpenAI library) ----
load_dotenv(Path(__file__).parent / ".env")   # reads GOOGLE_API_KEY
client = OpenAI(api_key=os.environ["GOOGLE_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = "gemini-flash-lite-latest"
