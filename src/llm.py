import logging

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def DeveloperMessage(content: str) -> ChatCompletionDeveloperMessageParam:
    """Used for o1/o3-mini models to set model behavior."""
    return {"role": "developer", "content": content}

def SystemMessage(content: str) -> ChatCompletionSystemMessageParam:
    """Standard role for gpt-4o/gpt-3.5 to set instructions."""
    return {"role": "system", "content": content}

def UserMessage(content: str) -> ChatCompletionUserMessageParam:
    """The human's input."""
    return {"role": "user", "content": content}

def AssistantMessage(content: str) -> ChatCompletionAssistantMessageParam:
    """The model's previous responses (for history)."""
    return {"role": "assistant", "content": content}

SYSTEM_MESSAGE = """You are a helpful assistant."""

class Client:
    def __init__(self, host: str = 'localhost', port: int = 8000, model_name: str = "org/qwen2.5-1m:7b"):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"
        self.model = model_name
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="token-not-needed",
            timeout=600
        )
        self.logger = logging.getLogger(__name__)

    def generate(self,  user_message: str, system_message: str=SYSTEM_MESSAGE):
        messages: list[ChatCompletionMessageParam] = [DeveloperMessage(system_message), UserMessage(user_message)]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise

    def test_connection(self):
        system_message = "You are a connection tester. If the user says 'ping', only respond with 'Hello'. Do not provide any other text."
        user_message = "ping"
        self.logger.info(f"Testing connection to {self.model}...")
        try:
            response = self.generate(
                user_message=user_message,
                system_message=system_message
            ).strip()

            if "hello" in response.lower():
                self.logger.info(f"Successfully connected to {self.base_url} with model {self.model}")
            else:
                self.logger.warning(f"Unexpected response format: {response}")
        except Exception as e:
            self.logger.critical(f"Connection test failed for {self.base_url}: {e}")
            raise ConnectionError(f"Could not connect to LLM server: {e}")


if __name__ == "__main__":
    #llm = Client(host="localhost", port=11434, model_name="mistral:latest")
    #llm = Client(host="localhost", port=1234, model_name="mistralai_devstral-small-2-24b-instruct-2512") # developer role
    llm = Client(host="172.18.11.146", port=8080, model_name="mistralai/Devstral-Small-2-24B-Instruct-2512") # system role
    llm.test_connection()