import json
import logging
import os
import time
import random
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam
)

from rich.console import Console
from rich.table import Table

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
load_dotenv()

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
    def __init__(self, host: str = 'localhost', port: int = 8000, base_url: str = None, model_name: str = "mistralai_devstral-small-2-24b-instruct-2512", api_key: str = None):
        self.logger = logging.getLogger(__name__)
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = f"http://{host}:{port}/v1"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "token-not-needed"
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=600
        )
        self.available_models = self.get_models()
        self.model = self.validate_model(model_name)

    def generate(self,  user_message: str, system_message: str=SYSTEM_MESSAGE, system_role: str = "system"):
        if system_role == "developer":
            developer_message = DeveloperMessage(system_message)
        else:
            developer_message = SystemMessage(system_message)
        messages: list[ChatCompletionMessageParam] = [developer_message, UserMessage(user_message)]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    def generate_with_retries(self,  user_message: str, system_message: str=SYSTEM_MESSAGE, system_role: str = "system", stop: list[str] | None = None):
        if system_role == "developer":
            developer_message = DeveloperMessage(system_message)
        else:
            developer_message = SystemMessage(system_message)
        messages: list[ChatCompletionMessageParam] = [developer_message, UserMessage(user_message)]


        max_retries = 6
        backoff_schedule_s = [5 * 60, 10 * 60] + [15 * 60] * (max_retries - 2)
        retry_after_cap_s = 15 * 60
        for attempt in range(1, max_retries + 1):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    top_p=1,
                    max_tokens=256,
                )
                if stop:
                    kwargs["stop"] = stop
                response = self.client.with_options(max_retries=1).chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except RateLimitError as e:
                sleep_s = backoff_schedule_s[attempt - 1]
                retry_after = None
                try:
                    headers = getattr(getattr(e, "response", None), "headers", None)
                    if headers:
                        ra = headers.get("Retry-After") or headers.get("retry-after")
                        if ra:
                            retry_after = float(ra)
                except Exception:
                    retry_after = None

                if retry_after is not None:
                    retry_after = max(0.0, min(retry_after, retry_after_cap_s))

                    sleep_s = min(sleep_s, retry_after)
                sleep_s += random.uniform(0, 1.5)  # jitter
                self.logger.warning(
                    f"429 rate limit (attempt {attempt}/{max_retries}). Sleeping {sleep_s:.1f}s. Error: {e}"
                )
                time.sleep(sleep_s)
                continue

            except (APITimeoutError, APIError) as e:
                if attempt == max_retries:
                    self.logger.error(f"Generation failed after {max_retries} attempts: {e}")
                    raise
                sleep_s = backoff_schedule_s[attempt - 1] + random.uniform(0, 1.5)
                self.logger.warning(f"Transient error (attempt {attempt}/{max_retries}). Sleeping {sleep_s:.1f}s. Error: {e}")
                time.sleep(sleep_s)
                continue

            except Exception as e:
                self.logger.error(f"Generation failed (non-retryable): {e}")
                raise
        return None
    def validate_model(self, requested_model: str) -> str:
        """
        Checks if the requested model exists.
        If yes, returns it.
        If no, returns the first available model and logs a warning.
        """
        if not self.available_models:
            self.logger.warning(
                f"Could not fetch models from {self.base_url}. Defaulting to requested '{requested_model}'.")
            return requested_model

        if requested_model in self.available_models:
            self.logger.info(f"Model '{requested_model}' confirmed available.")
            return requested_model
        else:
            fallback_model = self.available_models[0]
            self.logger.warning(
                f"Requested model '{requested_model}' NOT found. Switching to available model: '{fallback_model}'.")
            self.logger.info(f"Available Models:\n{json.dumps(self.available_models, indent=4)}")
            return fallback_model

    def get_models(self):
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []

    def test_connection(self):
        self.logger.info(f"Testing connection to {self.base_url}...")
        try:
            models = self.get_models()
            if not models:
                raise RuntimeError("Connected, but models list is empty.")
            self.logger.info(f"Successfully connected to {self.base_url}. Model '{self.model}' is available.")
        except Exception as e:
            self.logger.critical(f"Connection test failed for {self.base_url}: {e}")
            raise ConnectionError(f"Could not connect to LLM server: {e}")


if __name__ == "__main__":
    #llm = Client(host="localhost", port=11434, model_name="mistral:latest")
    #llm = Client(host="localhost", port=1234, model_name="mistralai_devstral-small-2-24b-instruct-2512") # developer role
    #llm = Client(host="172.18.11.146", port=8080, model_name="mistralai/Devstral-Small-2-24B-Instruct-2512") # system role
    llm = Client(base_url="https://chat-ai.academiccloud.de/v1" , model_name="mistral-large-3-675b-instruct-2512") # system role
    llm.test_connection()
