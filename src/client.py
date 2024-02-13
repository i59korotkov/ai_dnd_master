from typing import Optional, Union
import json

from openai import OpenAI


class Client:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo-0125",
        temperature: float = 0.7,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.model = model
        self.temperature = temperature

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )


    def get_response(
        self,
        messages: list,
        functions: Optional[list] = None,
        function_call = "auto",
        temperature: Optional[float] = None,
    ) -> Union[str, dict]:
        if functions is None:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature if temperature is None else temperature,
                messages=messages,
            ).choices[0]
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature if temperature is None else temperature,
                messages=messages,
                functions=functions,
                function_call=function_call,
            ).choices[0]

        # if response.finish_reason == "function_call": # For some reason, this doesn't work anymore
        if response.message.function_call is not None:
            return {
                "name": response.message.function_call.name,
                "arguments": json.loads(response.message.function_call.arguments),
            }
        else:
            return response.message.content
