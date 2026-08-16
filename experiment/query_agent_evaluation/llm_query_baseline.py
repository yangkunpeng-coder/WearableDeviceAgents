import os
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)
from textwrap import dedent
from utils.file_tool import json_read_to_dict
from utils.convert_data import convert_user_data

class LLMQueryBaseline:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com")

    def run_llm_query_baseline(self, question, file_dict):
        base_info, data = convert_user_data(file_dict)
        prompt = f"""
        Based on your "base_info", the "data" collected by the device over three months, and answer the "question".
        Note:1、answer in English；
        {base_info}
        {data}
        # question
        {question}
        """
        prompt = dedent(prompt)
        messages = [
            ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant"),
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]
        response = self.client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
            temperature=1.0,
        )
        output = response.choices[0].message.content
        tokens_used = response.usage
        # print(f"提示词 tokens: {tokens_used.prompt_tokens}")
        # print(f"生成内容 tokens: {tokens_used.completion_tokens}")
        # print(f"总计 tokens: {tokens_used.total_tokens}")
        return output, tokens_used.total_tokens

if __name__ == '__main__':
    llm_baseline = LLMQueryBaseline()
    role_data_file_path = '../../data/roles_data_en_10000/role_data_1.json'
    file_dict = json_read_to_dict(role_data_file_path)
    rewrite_task = "ow many atrial fibrillation episodes were detected from 2026-03-28 to 2026-03-30"
    llm_baseline_answer = llm_baseline.run_llm_query_baseline(rewrite_task, file_dict)
    print(llm_baseline_answer)
