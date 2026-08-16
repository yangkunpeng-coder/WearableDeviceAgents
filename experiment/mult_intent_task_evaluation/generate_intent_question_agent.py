import os
import random
from textwrap import dedent
from utils.file_tool import json_read_to_dict, dict_save_to_json
from utils.convert_data import convert_user_data
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)

def generate_mult_intents_agent(base_info, data, intents):
    prompt = f"""
    You are a user of a smart wearable device. Your basic information is 'base_info', and the data collected by the smart wearable device within one month is 'datas'. 
    Assume you have a smart wearable device assistant that can answer your questions. 
    Now, based on multiple intentions 'intents', you are required to ask the smart wearable device assistant questions that contain multiple intentions. 
    Note:1、answer in English.2、For query intentions, the questions should ask for data that can be found in 'data'.";3、The meaning of these intentions is 'the meaning of intent type.
    {base_info}
    {data}
    # intents type
    {intents}
    # the meaning of intent type
    1. query: refers to the user querying information, such as exercise records, health records, personal information;
    2. advice: refers to the user needing advice, such as exercise prescription, dietary prescription, health recommendations;
    3. analyse: refers to the need to analyze user data;
    """
    prompt = dedent(prompt)
    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")
    messages = [
        ChatCompletionSystemMessageParam(role="system", content="You are a user of a smart wearable device. "),
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        temperature=1.5
    )
    return response.choices[0].message.content

def generate_intents(min_count=3, max_count=8):
    intent_types = ['query', 'advice', 'analyse']
    num_intents = random.randint(min_count, max_count)
    return [random.choice(intent_types) for _ in range(num_intents)]

def generate_mult_intents_question():
    question_dict_s = json_read_to_dict('question_dict_s.json')
    for role_i in range(30, 31, 1):
        question_dict = {'question': [],
                         'intents': []}
        role_data_file_path = '../../data/roles_data_en_10000/role_data_{}.json'.format(role_i)
        data_dict = json_read_to_dict(role_data_file_path)
        question_dict['role_id'] = data_dict['role']['name_id']
        base_info, data = convert_user_data(data_dict)
        # intents = str(generate_intents())
        intents = ['{}, {}'.format('query', 'query'),
                   '{}, {}'.format('query', 'advice'),
                   '{}, {}'.format('query', 'analyse'),
                   '{}, {}, {}'.format('query', 'query', 'query'),
                   '{}, {}, {}'.format('query', 'query', 'advice'),
                   '{}, {}, {}'.format('query', 'analyse', 'advice')]
        for intent in intents:
            question_str = generate_mult_intents_agent(base_info, data, intent)
            question_dict['question'].append(question_str)
            question_dict['intents'].append(intent)
        question_dict_s.append(question_dict)
    dict_save_to_json(question_dict_s, 'question_dict_s.json')

if __name__ == '__main__':
    generate_mult_intents_question()
