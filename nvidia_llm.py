from openai import OpenAI
from config import NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_MODEL


class NvidiaLLM:
    def __init__(self):
        if not NVIDIA_API_KEY:
            raise ValueError(
                "NVIDIA_API_KEY is missing. Add it to your .env file."
            )

        self.client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY
        )

    def generate(self, messages, temperature=0.1, max_tokens=1500):
        response = self.client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=messages,
            temperature=temperature,
            top_p=0.7,
            max_tokens=max_tokens,
            stream=False
        )

        return response.choices[0].message.content
