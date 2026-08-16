import json
from textwrap import dedent
from datetime import datetime
from pydantic import BaseModel, Field
from utils.convert_data import convert_user_data
from brain_s.deepseek_v4_pro import DeepseekV4ProBrain
from base_class.agent_base import AgentBase


class QueryEngine:
    def __init__(self, role_data_dict):
        self.file_dict = role_data_dict

    def get_current_datetime(self):
        return  self.file_dict['data_s'][-1]['datetime']

    def select_convert_user_data(self, datetime_str_s):
        datetime_s = [datetime.strptime(datetime_str, "%Y-%m-%d") for datetime_str in datetime_str_s]
        data_dict = self.file_dict
        role_dict = data_dict['role']
        base_info = f"""
        # demographic_information
        {role_dict['demographic_information']}
        # lifestyle
        {role_dict['lifestyle']}
        # medical_history
        {role_dict['medical_history']}
    """
        data_str = ''
        for day_data in data_dict['data_s']:
            if datetime.strptime(day_data['datetime'], "%Y-%m-%d") in datetime_s:
                data_str += '## {}\n'.format(day_data['datetime'])
                data_str += '### sport data (Refers to the full-day exercise record data.)\n'
                for sport_dict in day_data['sport_s']:
                    data_str += 'Start at {} and end at {}, do {} minutes of {} exercise, consume {} kilocalories.\n'.format(
                        sport_dict['start_time'],
                        sport_dict['end_time'],
                        sport_dict['sport_time'],
                        sport_dict['sport_type'],
                        sport_dict['calorie'])
                data_str += '### health data (Refers to the full-day health monitoring data.)\n'
                for health_dict in day_data['health_s']:
                    for af in health_dict['atrial_fibrillation']:
                        data_str += '{} Detected {} time atrial fibrillation.\n'.format(af['record_time'], af['value'])
                    for ph in health_dict['premature_heartbeat']:
                        data_str += '{} Detected {} time premature heartbeat.\n'.format(ph['record_time'], ph['value'])
                    for sa in health_dict['sleep_apnea']:
                        data_str += '{} Detected {} time sleep apnea.\n'.format(sa['record_time'], sa['value'])
                    data_str += ('Sleep onset time is {}, total sleep duration is {} minutes, and sleep score is {}.\n'
                                 .format(health_dict['sleep_stage']['start_time'],
                                         health_dict['sleep_stage']['sleep_time'],
                                         health_dict['sleep_stage']['sleep_scope']))
                data_str += '### activity data (Refers to the full-day activity data record, which includes full-day step count, full-day calories burned, and full-day sedentary count.)\n'
                for activity_dict in day_data['activity_s']:
                    all_day_step_count = activity_dict['all_day_step_count']
                    all_day_sitting_num = activity_dict['all_day_sitting_num']
                    all_day_calorie = activity_dict['all_day_calorie']
                    data_str += (
                        'the entire day saw a total of {} steps taken, {} sedentary periods, and a daily calorie consumption of {} kilocalories.\n'
                        .format(all_day_step_count, all_day_sitting_num, all_day_calorie))
        data = f"""
    # datas
    {data_str}
    """
        base_info = dedent(base_info)
        data = dedent(data)
        return base_info, data


class DatetimeDictForm(BaseModel):
    datetime_dict: list[str] = Field(description='The date list of the query. the date format is %Y-%m-%d')


class QueryAgent(AgentBase):
    def __init__(self, brain):
        super().__init__(brain)

    def run_agent(self, **kwargs):
        self.internal_inout['communications'] = []
        """
        Execute the agent task.
        Keyword Args:
            question (str): User question, required.
            role_data_dict (dict): Role data dictionary, defaults to {}.
        """
        question = kwargs.get("question")
        if question is None:
            raise TypeError("run_agent() missing required keyword-only argument 'question'")
        role_data_dict = kwargs.get("role_data_dict")
        if role_data_dict is None:
            raise TypeError("run_agent() missing required keyword-only argument 'role_data_dict'")

        tokens_used = 0

        # query_engine
        query_engine = QueryEngine(role_data_dict=role_data_dict)

        current_datetime = query_engine.get_current_datetime()
        prompt = QueryAgent.datetime_prompt(form=DatetimeDictForm.model_json_schema(), question=question, current_datetime=current_datetime)
        sys_prompt = "You are a query agent_s responsible for retrieving results."
        output_dict, total_tokens = self.brain(sys_prompt=sys_prompt,
                   user_prompt=prompt, formating=DatetimeDictForm)
        datetime_str_s = output_dict.datetime_dict
        tokens_used += total_tokens
        self.internal_inout['communications'].append({'communication_name': 'query_datetime',
                                                     'sys_prompt': sys_prompt,
                                                     'user_prompt': prompt,
                                                     'output_dict': output_dict.model_dump_json()})
        if len(datetime_str_s) == 0:
            base_info, data = convert_user_data(data_dict=role_data_dict)
        else:
            base_info, data = query_engine.select_convert_user_data(datetime_str_s)

        # agent_s analyse
        prompt = QueryAgent.query_prompt(base_info, data, question)
        sys_prompt = "You are a helpful assistant"
        output, total_tokens = self.brain(sys_prompt=sys_prompt,
                   user_prompt=prompt)
        tokens_used += total_tokens
        self.internal_inout['communications'].append({'communication_name': 'query_summary',
                                                    'sys_prompt': sys_prompt,
                                                    'user_prompt': prompt,
                                                    'output_dict': output})
        
        return output, tokens_used

    @staticmethod
    def datetime_prompt(form, question, current_datetime):
        prompt = f"""
Your task is to break down the "question" into a series of datetime list.
Note:1、Output must be in JSON format without any explanation or additional text.
# current date
current date is {current_datetime}
# question
{question}
# Format of the generated result
{form}
"""
        prompt = dedent(prompt)
        return prompt

    @staticmethod
    def query_prompt(base_info, data, question):
        prompt = f"""
Based on your "base_info", the "data" collected by the device, and answer the "question".
Note:1、answer in English;
{base_info}
{data}
# question
{question}
"""
        prompt = dedent(prompt)
        return prompt

if __name__ == '__main__':

    is_deepseek = True
    deepseek_v4_pro_brain = DeepseekV4ProBrain()
    # qwen3_5_08B = Qwen3_5Brain(model_name='qwen3.5_0.8B')
    query_agent = QueryAgent(deepseek_v4_pro_brain)
    role_data_file_path = '../data/roles_data_en_10000/role_data_1.json'
    with open(role_data_file_path, 'r', encoding='utf-8') as f:
        file_dict = json.load(f)
    content_str = "What are the daily step counts, average step count, maximum step count, and minimum step count for the past week? And what is the trend?"
    json_form, tokens = query_agent.run_agent(question=content_str, role_data_dict=file_dict)
    internal_inout = query_agent.get_internal_brain_inout()
    print(json_form)
    print(internal_inout)
    '''
    internal_inout = {'agent_name': 'QueryAgent', 'communications': [{'communication_name': 'query_datetime', 'sys_prompt': 'You are a query agent_s responsible for retrieving results.', 'user_prompt': '\nYour task is to break down the "question" into a series of datetime list.\nNote:1、Output must be in JSON format without any explanation or additional text.\n# current date\ncurrent date is 2026-03-30\n# question\nQuery the user\'s daily step count for each day over the past week.\n# Format of the generated result\n{\'properties\': {\'datetime_dict\': {\'description\': \'The date list of the query. the date format is %Y-%m-%d\', \'items\': {\'type\': \'string\'}, \'title\': \'Datetime Dict\', \'type\': \'array\'}}, \'required\': [\'datetime_dict\'], \'title\': \'DatetimeDictForm\', \'type\': \'object\'}\n', 'output_dict': '{"datetime_dict":["2026-03-24","2026-03-25","2026-03-26","2026-03-27","2026-03-28","2026-03-29","2026-03-30"]}'}, {'communication_name': 'query_summary', 'sys_prompt': 'You are a helpful assistant', 'user_prompt': '\nBased on your "base_info", the "data" collected by the device, and answer the "question".\nNote:1、answer in English;\n\n# demographic_information\nMale, 25 years old, urban professional, college graduate, single\n# lifestyle\nActive lifestyle, works in tech industry, gym 4 times/week, moderate caffeine consumption, occasional social drinking\n# medical_history\nNo significant medical conditions, normal blood pressure, occasional stress-related headaches\n\n\n    # datas\n    ## 2026-03-24\n### sport data (Refers to the full-day exercise record data.)\nStart at 18:00:00 and end at 19:00:00, do 60 minutes of Swimming exercise, consume 440.0 kilocalories.\n### health data (Refers to the full-day health monitoring data.)\n09:30:00 Detected 0 time atrial fibrillation.\n20:00:00 Detected 0 time atrial fibrillation.\n10:30:00 Detected 0 time premature heartbeat.\nSleep onset time is 23:15:00, total sleep duration is 450 minutes, and sleep score is 78.\n### activity data (Refers to the full-day activity data record, which includes full-day step count, full-day calories burned, and full-day sedentary count.)\nthe entire day saw a total of 9000 steps taken, 6 sedentary periods, and a daily calorie consumption of 2080.0 kilocalories.\n## 2026-03-25\n### sport data (Refers to the full-day exercise record data.)\nStart at 12:00:00 and end at 12:45:00, do 45 minutes of Cycling exercise, consume 330.0 kilocalories.\n### health data (Refers to the full-day health monitoring data.)\n14:30:00 Detected 0 time atrial fibrillation.\n08:00:00 Detected 0 time premature heartbeat.\n16:00:00 Detected 0 time premature heartbeat.\nSleep onset time is 23:40:00, total sleep duration is 440 minutes, and sleep score is 73.\n### activity data (Refers to the full-day activity data record, which includes full-day step count, full-day calories burned, and full-day sedentary count.)\nthe entire day saw a total of 7700 steps taken, 7 sedentary periods, and a daily calorie consumption of 1820.0 kilocalories.\n## 2026-03-26\n### sport data (Refers to the full-day exercise record data.)\nStart at 18:30:00 and end at 19:30:00, do 60 minutes of Weight Training exercise, consume 380.0 kilocalories.\n### health data (Refers to the full-day health monitoring data.)\n08:30:00 Detected 0 time atrial fibrillation.\n17:30:00 Detected 0 time atrial fibrillation.\n09:30:00 Detected 0 time premature heartbeat.\n20:00:00 Detected 1 time premature heartbeat.\nSleep onset time is 23:10:00, total sleep duration is 440 minutes, and sleep score is 77.\n### activity data (Refers to the full-day activity data record, which includes full-day step count, full-day calories burned, and full-day sedentary count.)\nthe entire day saw a total of 8500 steps taken, 7 sedentary periods, and a daily calorie consumption of 1960.0 kilocalories.\n## 2026-03-27\n### sport data (Refers to the full-day exercise record data.)\nStart at 07:00:00 and end at 07:40:00, do 40 minutes of Running exercise, consume 290.0 kilocalories.\n### health data (Refers to the full-day health monitoring data.)\n10:00:00 Detected 0 time atrial fibrillation.\n08:00:00 Detected 0 time premature heartbeat.\n12:30:00 Detected 0 time premature heartbeat.\nSleep onset time is 23:20:00, total sleep duration is 450 minutes, and sleep score is 83.\n### activity data (Refers to the full-day activity data record, which includes full-day step count, full-day calories burned, and full-day sedentary count.)\nthe entire day saw a total of 9800 steps taken, 5 sedentary periods, and a daily calorie consumption of 2120.0 kilocalories.\n## 2026-03-28\n### sport data (Refers to the full-day exercise record data.)\n### health data (Refers to the full-day health monitoring data.)\n12:00:00 Detected 0 time atrial fibrillation.\n14:30:00 Detected 0 time atrial fibrillation.\n11:00:00 Detected 0 time premature heartbeat.\nSleep onset time is 00:30:00, total sleep duration is 510 minutes, and sleep score is 89.\n### activity data (Refers to the full-day activity data record, which includes full-day step count, full-day calories burned, and full-day sedentary count.)\nthe entire day saw a total of 5200 steps taken, 10 sedentary periods, and a daily calorie consumption of 1600.0 kilocalories.\n## 2026-03-29\n### sport data (Refers to the full-day exercise record data.)\nStart at 16:00:00 and end at 17:00:00, do 60 minutes of Tennis exercise, consume 420.0 kilocalories.\n### health data (Refers to the full-day health monitoring data.)\n09:00:00 Detected 0 time atrial fibrillation.\n19:00:00 Detected 0 time atrial fibrillation.\n08:30:00 Detected 0 time premature heartbeat.\n17:30:00 Detected 0 time premature heartbeat.\nSleep onset time is 23:00:00, total sleep duration is 450 minutes, and sleep score is 82.\n### activity data (Refers to the full-day activity data record, which includes full-day step count, full-day calories burned, and full-day sedentary count.)\nthe entire day saw a total of 12000 steps taken, 5 sedentary periods, and a daily calorie consumption of 2400.0 kilocalories.\n## 2026-03-30\n### sport data (Refers to the full-day exercise record data.)\nStart at 07:00:00 and end at 07:45:00, do 45 minutes of Running exercise, consume 330.0 kilocalories.\n### health data (Refers to the full-day health monitoring data.)\n08:30:00 Detected 0 time atrial fibrillation.\n16:30:00 Detected 0 time atrial fibrillation.\n07:30:00 Detected 0 time premature heartbeat.\nSleep onset time is 23:15:00, total sleep duration is 450 minutes, and sleep score is 80.\n### activity data (Refers to the full-day activity data record, which includes full-day step count, full-day calories burned, and full-day sedentary count.)\nthe entire day saw a total of 10500 steps taken, 5 sedentary periods, and a daily calorie consumption of 2250.0 kilocalories.\n\n\n# question\nQuery the user\'s daily step count for each day over the past week.\n', 'output_dict': 'Here are the daily step counts over the past week:\n\n- 2026-03-24: 9,000 steps\n- 2026-03-25: 7,700 steps\n- 2026-03-26: 8,500 steps\n- 2026-03-27: 9,800 steps\n- 2026-03-28: 5,200 steps\n- 2026-03-29: 12,000 steps\n- 2026-03-30: 10,500 steps'}]}
    print(internal_inout['communications'][1]['output_dict'])
    '''