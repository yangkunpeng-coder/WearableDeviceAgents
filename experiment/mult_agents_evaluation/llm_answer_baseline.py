import os
import json
from textwrap import dedent
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)
from utils.convert_data import convert_user_data


def llm_baseline_manager_prompt(question, file_dict):
    base_info, data = convert_user_data(file_dict)
    answer_summary = '# question:\n{}\n'.format(question)
    answer_summary += base_info
    answer_summary += data
    prompt = f"""
Based on your "base_info", the "data" collected by the device over three months, answer "#question:{question}", requiring accurate, complete, and logical responses.
Note:1、Output must be without any explanation or additional text;2、answer in English;3、Please answer in Markdown format, using # for main sections, ## for subheadings, - for list items, and 1. for steps.
4、title is #Answer
{answer_summary}
    """
    prompt = dedent(prompt)
    return prompt

class LlmBaselineManagerAgent:
    # model
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com")

    def run_llm_baseline_manager_agent(self, role_data_dict, content):
        tokens_used = 0

        # aggregate
        co_prompt = llm_baseline_manager_prompt(question=content, file_dict=role_data_dict)
        messages = [
            ChatCompletionSystemMessageParam(role="system", content="You are an aggregate agent_s, responsible for answering the question of smartwatch users."),
            ChatCompletionUserMessageParam(role="user", content=co_prompt),
        ]
        print(co_prompt)
        response = self.client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
            temperature=1.0,
        )
        co_output = response.choices[0].message.content
        tokens_used += response.usage.total_tokens
        answer_dict_s = None
        return co_output, tokens_used, answer_dict_s


def test_manager_agent():
    # "Based on my data, what were my total steps and sleep scores on March 1 and March 15, and can you analyze how my exercise type on those days influenced my sleep? Finally, advise me on how to improve my sleep quality based on this analysis."
    content = "What was my sleep score, total steps, and the type of exercise I did on March 15, 2026?"
    role_data_path = '../../data/roles_data_en_10000/role_data_1.json'
    with open(role_data_path, 'r', encoding='utf-8') as f:
        role_data = json.load(f)
    m_agent = LlmBaselineManagerAgent()
    co_output, tokens_used, answer_dict_s = m_agent.run_llm_baseline_manager_agent(role_data, content)
    print('question:{}'.format(content))
    print('tokens_used:{}'.format(tokens_used))
    print('co_output:\n{}'.format(co_output))


if __name__ == '__main__':
    test_manager_agent()
