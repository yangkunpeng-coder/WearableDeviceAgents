import asyncio
import os
import glob
from datetime import datetime
from textwrap import dedent
from typing import Literal
import multiprocessing
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from pydantic import BaseModel, Field
from agent_config import get_model
from utils.file_tool import json_read_to_dict, dict_save_to_json


def role_sport_prompt(form, content, start_datetime, end_datetime):
    delta = datetime.strptime(end_datetime, '%Y-%m-%d') - datetime.strptime(start_datetime, '%Y-%m-%d')
    day_num = delta.days
    prompt = f"""
    generating {day_num} days's records from {start_datetime} to {end_datetime} based on "Basic information of the simulated character". 
    The output must strictly follow the specified format.
    ### Basic information of the simulated character.
    Demographic information: {content['demographic_information']}
    Lifestyle: {content['lifestyle']}
    Medical history: {content['medical_history']}
    ### Format of the generated result
    {form}
    """
    prompt = dedent(prompt)
    return prompt


class RoleSportForm(BaseModel):  # 运动
    start_time: str = Field(description="Exercise start time, format %H:%M:%S")
    end_time: str = Field(description="Exercise end time, format %H:%M:%S")
    sport_type: str = Field(description="Exercise type")
    sport_time: int = Field(description="Exercise duration, unit is minutes, the result should be just the number without the unit.")
    calorie: float = Field(description="Calorie")


class RoleAtrialFibrillationForm(BaseModel):  # 房颤
    record_time: str = Field(description="Time of atrial fibrillation occurrence record, format %H:%M:%S")
    value: int = Field(description="If atrial fibrillation is detected, the value is 1; if not detected, the value is 0.")


class RolePrematureHeartbeatForm(BaseModel):  # 早搏
    record_time: str = Field(description="Time of premature beat record, format %H:%M:%S")
    value: Literal[0, 1] = Field(description="1 if a premature beat is detected, otherwise 0.")


class RoleBloodOxygenForm(BaseModel):  # 血氧饱和度
    record_time: str = Field(description="Time of blood oxygen recording, format %H:%M:%S")
    value: int = Field(description="If blood oxygen saturation is measured, output the blood oxygen saturation value. At high altitudes, blood oxygen levels may decrease; during sleep apnea, blood oxygen levels may also decrease.")


class RoleSleepApneaForm(BaseModel):  # 睡眠呼吸暂停
    record_time: str = Field(description="Time of sleep apnea occurrence record, format %H:%M:%S")
    value: Literal[0, 1] = Field(description="If sleep apnea is detected, the value is 1; if not detected, the value is 0.")


class RoleSleepForm(BaseModel):  # 睡眠
    start_time: str = Field(description="Sleep onset time, format %H:%M:%S")
    end_time: str = Field(description="Wake-up time (time of awakening), format %H:%M:%S")
    sleep_time: int = Field(description="Sleep duration, in minutes.")
    sleep_scope: int = Field(description="Sleep quality score, with a maximum of 100 points.")


class RoleHealthForm(BaseModel):  # 健康
    atrial_fibrillation: list[RoleAtrialFibrillationForm] = Field(description="The smartwatch monitors atrial fibrillation daily, with random detection triggered manually by the user, and it can be performed multiple times a day.")
    premature_heartbeat: list[RolePrematureHeartbeatForm] = Field(description="The smartwatch monitors premature beats daily, with random detection triggered manually by the user, and it can be performed multiple times a day.")
    blood_oxygen: list[RoleBloodOxygenForm] = Field(description="The smartwatch monitors blood oxygen saturation daily, with random detection triggered manually by the user, and it can be performed multiple times a day.")
    sleep_apnea: list[RoleSleepApneaForm] = Field(description="If the user has sleep apnea, it may be detected by the smartwatch during the night, and it can occur once or multiple times per night.")
    sleep_stage: RoleSleepForm = Field(description="Sleep data monitored by the smartwatch.")


class RoleActivityForm(BaseModel):  # 日常活动
    all_day_step_count: int = Field(description="Daily step count")
    all_day_sitting_num: int = Field(description="Daily sedentary count")
    all_day_calorie: float = Field(description="Daily calories")


class RoleDataForm(BaseModel):  # 角色数据
    datetime: str = Field(description="Recorded date and time, format %Y-%m-%d")
    sport_s: list[RoleSportForm] = Field(description="List of exercises for the day, which can contain multiple items. If the watch is not worn or no exercise is performed, it will be 0 items.")
    health_s: list[RoleHealthForm] = Field(description="List of health records for the day, which can contain multiple items. If the watch is not worn, there will be no records for that day, resulting in 0 items.")
    activity_s: list[RoleActivityForm] = Field(description="List of all-day records for the day, which can contain multiple items including the number of sedentary periods (the number of times sitting for one hour), daily step count, daily calories, etc. If the watch is not worn, there will be no records for that day, resulting in 0 items.")


class DataListForm(BaseModel):
    data_s: list[RoleDataForm] = Field(description="List of data recorded by the user using a smartwatch.")


async def generate_sport_agent(role, start_datetime, end_datetime, model_type='deepseek'):

    model, chat_form = get_model(model_type, 1.5)

    prompt = role_sport_prompt(DataListForm.model_json_schema(), role, start_datetime, end_datetime)
    toolkit = Toolkit()
    agent = ReActAgent(
        name='data simulater',
        sys_prompt='You are a data simulator that generates simulated smartwatch data for users.',
        model=model,
        formatter=chat_form,
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    msg = Msg("user", prompt, "user")
    try:
        response = await agent(msg,
                               structured_model=DataListForm)

        if response.metadata is not None:
            json_form = response.metadata
            json_form['role'] = role
        else:
            json_form = ''
        return json_form
    except RuntimeError:
        print("error!")
        json_form = ''
        return json_form
    except AttributeError:
        print("error!")
        json_form = ''
        return json_form

def generate_role_data(role, start_datetime, end_datetime):
    json_form = asyncio.run(generate_sport_agent(role, start_datetime, end_datetime))
    if 'data_s' in json_form:
        dict_save_to_json(json_form, '../data/roles_data_en_10000/role_data_{}.json'.format(role['name_id']))


def check_generate_role_data():
    from utils.file_tool import json_read_to_dict
    for role_i in range(1, 1001, 1):
        role_data_file_path = '../data/roles_data_en_10000/role_data_{}.json'.format(role_i)
        file_dict = json_read_to_dict(role_data_file_path)
        if 'data_s' not in file_dict:
            print('format error:{}'.format(role_i))
            continue
        datetime_str_s = []
        for data_dict  in file_dict['data_s']:
            current_datetime = data_dict['datetime']
            datetime_str_s.append(current_datetime)
        total_days = len(datetime_str_s)
        if total_days < 20:
            print('day num error:{}'.format(role_i))
            continue

def generate_batch_role_data():
    role_s = json_read_to_dict('../data/role_list_en_10000.json')
    start_i = 990
    role_s['roles'] = role_s['roles'][start_i:]
    total_num = 1000 - start_i
    per_time = 10
    epoch_total_num = int(total_num / per_time)
    for i in range(epoch_total_num):
        process_s = []
        for epoch in range(per_time):
            base_i = i * per_time + epoch
            sub_process = multiprocessing.Process(target=generate_role_data, args=(role_s['roles'][base_i], '2026-3-01', '2026-3-30', ))
            process_s.append(sub_process)
        for sub_process in process_s:
            sub_process.start()
        for sub_process in process_s:
            sub_process.join()
    print('All processes completed.')

def generate_error_batch_role_data(i_s):
    process_s = []
    for i in i_s:
        role_s = json_read_to_dict('../data/role_list_en_10000.json')
        sub_process = multiprocessing.Process(target=generate_role_data,
                                              args=(role_s['roles'][i - 1], '2026-3-01', '2026-3-30',))
        process_s.append(sub_process)
    for sub_process in process_s:
        sub_process.start()
    for sub_process in process_s:
        sub_process.join()

def regenerate_error_data():
    role_s = json_read_to_dict('../data/role_list_en_10000.json')
    data_dir = '../data/roles_data_en_10000/'
    pattern = os.path.join(data_dir, '*.json')
    json_files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
    process_s = []
    for i in range(1, 1001, 1):
        data_path = data_dir + 'role_data_{}'.format(i) + '.json'
        data_dict = json_read_to_dict(data_path)
        datetime_str_s = []
        for d  in data_dict['data_s']:
            current_datetime = d['datetime']
            datetime_str_s.append(current_datetime)
        total_days = len(datetime_str_s)
        if total_days < 20:
            print(i)
            sub_process = multiprocessing.Process(target=generate_role_data, args=(role_s['roles'][i - 1], '2026-3-01', '2026-3-30',))
            process_s.append(sub_process)
        if len(process_s) >= 20:
            for sub_process in process_s:
                sub_process.start()
            for sub_process in process_s:
                sub_process.join()
            process_s.clear()
    if len(process_s) > 0:
        for sub_process in process_s:
            sub_process.start()
        for sub_process in process_s:
            sub_process.join()
    print('All processes completed.')

if __name__ == '__main__':
    check_generate_role_data()
    generate_error_batch_role_data([1, 2])
