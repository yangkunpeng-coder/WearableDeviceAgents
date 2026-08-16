import os
from dataclasses import dataclass
from agentscope.model import DashScopeChatModel, OpenAIChatModel
from agentscope.formatter import DashScopeChatFormatter, OpenAIChatFormatter


@dataclass
class ModelConfig:
    LLM_name_qwen: str = 'qwen3-max'
    api_key_qwen: str = os.getenv('DASHSCOPE_API_KEY')
    LLM_name_deepseek: str = 'deepseek-chat'
    api_key_deepseek: str = os.getenv('DEEPSEEK_API_KEY')


def get_model(model_type='deepseek', temperature=1.5):
    if model_type == 'qwen':
        model = DashScopeChatModel(
            model_name=ModelConfig.LLM_name_qwen,
            api_key=ModelConfig.api_key_qwen,
            stream=False,
            enable_thinking=False,
            generate_kwargs={'temperature': temperature}
        )
        chat_form = DashScopeChatFormatter()
    elif model_type == 'deepseek':
        model = OpenAIChatModel(
            model_name=ModelConfig.LLM_name_deepseek,
            api_key=ModelConfig.api_key_deepseek,
            stream=False,
            generate_kwargs={'temperature': temperature},
            client_kwargs={"base_url": "https://api.deepseek.com/v1"}
            # client_args={"base_url": "https://api.deepseek.com"}
        )
        chat_form = OpenAIChatFormatter()
    else:
        raise TypeError("input deepseek、qwen")
    return model, chat_form
