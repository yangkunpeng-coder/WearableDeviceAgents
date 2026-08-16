import os
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    completion_create_params,
)

from base_class.brain_base import BrainBase

class DeepseekV4ProBrain(BrainBase):
    def __init__(self):
        self.brain = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com")

    def __call__(self, sys_prompt, user_prompt, formating=None, **kwargs):
        output_dict = None
        total_tokens = 0
        messages = [
            ChatCompletionSystemMessageParam(role="system", content=sys_prompt),
            ChatCompletionUserMessageParam(role="user", content=user_prompt),
        ]
        if formating is not None:
            response = self.brain.chat.completions.create(
                model="deepseek-v4-pro",
                messages=messages,
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
                temperature=1.0,
                response_format= completion_create_params.ResponseFormatJSONObject(type='json_object'),
            )
        else:
            response = self.brain.chat.completions.create(
                model="deepseek-v4-pro",
                messages=messages,
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
                temperature=1.0,
            )
        if response.usage is not None:
            total_tokens = response.usage.total_tokens
        output = response.choices[0].message.content
        if output is not None:
            if formating is not None:
                output_dict = formating.model_validate_json(output)
            else:
                output_dict = output
        return output_dict, total_tokens
    