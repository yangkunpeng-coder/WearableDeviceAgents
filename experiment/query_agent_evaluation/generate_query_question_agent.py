import os
import random
from datetime import datetime, timedelta
from textwrap import dedent
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)
from utils.file_tool import json_read_to_dict
from base_class.agent_base import AgentBase


def query_data_item(data_dict, data_type, datetime_s):
    result = 0
    datetime_s = [datetime.strptime(datetime_oneday, "%Y-%m-%d") for datetime_oneday in datetime_s]
    for data_dict in data_dict['data_s']:
        current_date_str = data_dict['datetime']
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        if current_date in datetime_s:
            if data_type == 'sport':
                result += len(data_dict['sport_s'])
            elif data_type == 'atrial_fibrillation':
                for health_dict in data_dict['health_s']:
                    af_s = health_dict['atrial_fibrillation']
                    for af in af_s:
                        result += af['value']
            elif data_type == 'premature_heartbeat':
                for health_dict in data_dict['health_s']:
                    p_s = health_dict['premature_heartbeat']
                    for p in p_s:
                        result += p['value']
            elif data_type == 'sleep_apnea':
                for health_dict in data_dict['health_s']:
                    sa_s = health_dict['sleep_apnea']
                    for sa in sa_s:
                        result += sa['value']
            elif data_type == 'all_day_step_count':
                for activity_dict in data_dict['activity_s']:
                    result += activity_dict['all_day_step_count']
            elif data_type == 'all_day_sitting_num':
                for activity_dict in data_dict['activity_s']:
                    result += activity_dict['all_day_sitting_num']
            elif data_type == 'all_day_calorie':
                for activity_dict in data_dict['activity_s']:
                    result += activity_dict['all_day_calorie']
            else:
                ...
    return result

def expand_date_range(date_range):
    start = datetime.strptime(date_range[0], '%Y-%m-%d')
    end = datetime.strptime(date_range[1], '%Y-%m-%d')
    delta = (end - start).days
    expanded = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(delta + 1)]
    return expanded


def random_date_type(datetime_str_s):
    type_choice = random.choice(['single', 'period', 'multiple'])
    start = datetime.strptime(datetime_str_s[0], '%Y-%m-%d')
    total_days = len(datetime_str_s)
    if total_days < 3:
        type_choice = 'single'
    total_ids = list(range(0, total_days))
    if type_choice == 'single':
        select_ids = random.sample(total_ids, 1)
        datetime_s = [start + timedelta(days=select_ids[0])]
        dates = [d.strftime('%Y-%m-%d') for d in datetime_s]
    elif type_choice == 'period':
        total_ids = list(range(0, total_days))
        select_ids = random.sample(total_ids, 2)
        offsets = sorted(select_ids)
        datetime_s = [start + timedelta(days=off) for off in offsets]
        dates = [d.strftime('%Y-%m-%d') for d in datetime_s]
    else:  # multiple
        min_n = 3
        if total_days < min_n:
            n = total_days
        else:
            n = random.randint(min_n, total_days)
        select_ids = random.sample(total_ids, n)
        offsets = sorted(select_ids)
        datetime_s = [start + timedelta(days=off) for off in offsets]
        dates = [d.strftime('%Y-%m-%d') for d in datetime_s]
    return type_choice, dates

def create_query_question():
    question_ch = [
        {'task': '在{}共做了多久次运动', 'data_type': 'sport'},
        {'task': '在{}共检测出多少次房颤', 'data_type': 'atrial_fibrillation'},
        {'task': '在{}共检测出多少次早搏', 'data_type': 'premature_heartbeat'},
        {'task': '在{}共检测出多少次睡眠呼吸暂停', 'data_type': 'sleep_apnea'},
        {'task': '在{}步数的总和', 'data_type': 'all_day_step_count'},
        {'task': '在{}卡路里的总和', 'data_type': 'all_day_calorie'},
        {'task': '在{}久坐次数的总和', 'data_type': 'all_day_sitting_num'},
    ]
    return question_ch

def create_advice_question(start_datetime, end_datetime):
    start_datetime_str = start_datetime
    end_datetime_str = end_datetime
    question_ch = [
        # query 1
        {'question': '我在{}做了多久次运动？'.format(start_datetime_str),
         'intents': [{'intent': 'query', 'tasks': ['查询在{}做了多久次运动'.format(start_datetime_str)], 'agents': ['query_agent']}]},
    ]
    return question_ch

def translate_language_prompt(intent, base_info):
    prompt = f"""
Based on the "user's basic information", rewrite "user's input" in English using the user's expression style, keeping the meaning unchanged.
Please note: "Choose one of the 1–8 data formats in 'Date Expressions' as the expression method."
# user's input
{intent}
# user's basic information
{base_info}
# Date Expressions
1. US written formats
   - Numeric: MM/DD/YYYY (e.g., 09/05/2026)
   - Full: Month Day, Year (e.g., September 5, 2026)
   - With ordinal (informal): September 5th, 2026
2. UK written formats
   - Numeric: DD/MM/YYYY (e.g., 05/09/2026)
   - Full: Day Month Year (e.g., 5 September 2026)
   - With ordinal: 5th September 2026
   - Formal: the 5th of September, 2026
3. International standard (ISO)
   - YYYY-MM-DD (e.g., 2026-09-05)
4. Spoken (US)
   - "September fifth" or "September the fifth"
   - With year: "September fifth, twenty twenty-six"
5. Spoken (UK)
   - "the fifth of September"
   - With year: "the fifth of September, twenty twenty-six"
6. Prepositions
   - Use "on" for a specific day: on September 5
   - Use "in" for month or year only: in September, in 2026
7. Reading years
   - 2010+: split into pairs: twenty twenty-six
   - 2000–2009: two thousand (and) ... (e.g., two thousand and nine)
   - 2000: two thousand
   - 1900: nineteen hundred (also possible)
8. Avoiding ambiguity
   - Avoid all-numeric dates (e.g., 02/03/2026) unless the region is known.
   - Spell out the month or use ISO format (YYYY-MM-DD).
    """
    prompt = dedent(prompt)
    return prompt

class ReWriteAgent(AgentBase):
    def __init__(self, brain):
        super().__init__(brain)

    def run_agent(self, **kwargs):
        self.internal_inout["communications"] = []
        """
        Execute the agent task.
        Keyword Args:
            content (str): User question, required.
            base_info (dict): Role data dictionary, defaults to {}.
        """
        content = kwargs.get("content")
        if content is None:
            raise TypeError(
                "run_agent() missing required keyword-only argument 'content'"
            )
        base_info = kwargs.get("base_info")
        if base_info is None:
            raise TypeError(
                "run_agent() missing required keyword-only argument 'base_info'"
            )
        tokens_used = 0
        prompt = translate_language_prompt(intent=content, base_info=base_info)
        sys_prompt = "You are a helpful assistant"
        output, total_tokens = self.brain(sys_prompt=sys_prompt,
                   user_prompt=prompt)
        tokens_used += total_tokens
        return output, tokens_used

async def translate_language_agent(content, base_info):
    prompt = translate_language_prompt(intent=content, base_info=base_info)

    client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")
    messages = [
        ChatCompletionSystemMessageParam(role="system", content="You are a helpful assistant"),
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
        temperature=1.0
    )
    output = response.choices[0].message.content
    return output

def test_generate_query_question():
    for role_i in range(1, 1001, 1):
        role_data_file_path = '../../data/roles_data_en_10000/role_data_{}.json'.format(role_i)
        file_dict = json_read_to_dict(role_data_file_path)
        datetime_str_s = []
        for data_dict  in file_dict['data_s']:
            current_datetime = data_dict['datetime']
            datetime_str_s.append(current_datetime)
        total_days = len(datetime_str_s)
        query_question_ch = create_query_question()
        question_s = random.choices(query_question_ch, k=10)
        for question in question_s:
            data_type = question['data_type']
            type_choice, dates = random_date_type(datetime_str_s)
            datetime_s = dates
            datetime_str = ''
            if type_choice == 'single':
                datetime_str = dates[0]
            elif type_choice == 'period':
                datetime_str = '{}至{}'.format(dates[0], dates[-1])
                datetime_s = expand_date_range(dates)
            elif type_choice == 'multiple':
                for di, date in enumerate(dates):
                    datetime_str += date
                    if di != len(dates) - 1:
                        datetime_str += '、'
                datetime_str += '这{}天'.format(len(dates))
            true_result = query_data_item(file_dict, data_type, datetime_s)
            print(datetime_str)
            print(true_result)

if __name__ == '__main__':
    ...
