import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class HelloAgentLLM:
    def __init__(
        self,
        model: str = None,
        apiKey: str = None,
        baseUrl: str = None,
        timeout: int = None,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.apiKey = apiKey or os.getenv("LLM_API_KEY")
        self.baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        self.client = OpenAI(
            api_key=self.apiKey,
            base_url=self.baseUrl,
            timeout=self.timeout,
        )

    def think(self, messages, temperature=0, stream=False):
        if not stream:
            try:
                resource = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=False,
                )
                return resource.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️  调用LLM API时发生错误: {e}")
                return None

        def generator():
            try:
                resource = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )

                for chunk in resource:
                    try:
                        choices = getattr(chunk, "choices", None)
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = getattr(choice, "delta", None)
                        if delta is None:
                            continue

                        content = getattr(delta, "content", None)
                        if isinstance(content, str) and content:
                            yield content
                    except Exception:
                        continue

            except Exception as e:
                print(f"⚠️  调用LLM API时发生错误: {e}")
                return

        return generator()

    def recognize_intent(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        return self.think(messages, temperature=0, stream=False)

    def generate_answer(self, prompt: str, stream=True):
        messages = [{"role": "user", "content": prompt}]
        return self.think(messages, temperature=0, stream=stream)