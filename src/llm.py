from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionUserMessageParam
)


def DeveloperMessage(content: str) -> ChatCompletionDeveloperMessageParam:
    return {"role": "system", "content": content}

def UserMessage(content: str) -> ChatCompletionUserMessageParam:
    return {"role": "user", "content": content}

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

    def generate(self,  user_message: str, system_message: str=SYSTEM_MESSAGE):
        messages: list[ChatCompletionMessageParam] = [DeveloperMessage(system_message), UserMessage(user_message)]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    #llm = Client(host="localhost", port=11434, model_name="mistral:latest")
    llm = Client(host="localhost", port=1234, model_name="mistralai_devstral-small-2-24b-instruct-2512") # developer role
    #llm = Client(host="172.18.11.146", port=8080, model_name="mistralai/Devstral-Small-2-24B-Instruct-2512") # system role
    print(llm.generate(
        system_message="You are a helpful assistant. Given Iris flower features (sepal length, sepal width, petal length, petal width), return only the predicted species name and nothing else.",
        user_message="A flower has a sepal length of 6.1cm, a sepal width of 3.0cm, a petal length of 4.9cm, and a petal width of 1.8cm. What is this species?"))
