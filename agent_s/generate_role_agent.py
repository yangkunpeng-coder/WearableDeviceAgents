import asyncio
from pathlib import Path
import multiprocessing
from textwrap import dedent
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from pydantic import BaseModel, Field
from agent_config import get_model
from utils.file_tool import dict_save_to_json, json_read_to_dict


def role_create_prompt(form, role_num):
    prompt = f"""
    You are a data generator that generates basic information of users who use smartwatches. You need to generate simulated information for {role_num} users from aspects such as demographic information, lifestyle, and health history, and strictly follow the output format of the generated results.
    It is required that the distribution across different age groups be even, and about 50% of the population have one or more of the following conditions: atrial fibrillation, premature beats, sleep apnea, or insufficient sleep duration.
    ### Format of the generated result
    {form}
    """
    prompt = dedent(prompt)
    return prompt


class RoleCreatorForm(BaseModel):
    name_id: int = Field(description="Unique ID number, non-repeating.")
    demographic_information: str = Field(description="Demographic information")
    lifestyle: str = Field(description="lifestyle")
    medical_history: str = Field(description="health history")


class RoleListForm(BaseModel):
    roles: list[RoleCreatorForm] = Field(description="List of roles of smartwatch users")


async def generate_role_agent(role_num, model_type='deepseek'):
    """
    :param role_num:
    :param model_type: deepseek、qwen
    :return:
    """
    model, chat_form = get_model('deepseek', 1.5)

    prompt = role_create_prompt(RoleListForm.model_json_schema(), role_num)
    toolkit = Toolkit()
    agent = ReActAgent(
        name='role creator',
        sys_prompt='You are a data generator that generates basic information of users who use smartwatches.',
        model=model,
        formatter=chat_form,
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    msg = Msg("user", prompt, "user")
    try:
        response = await agent(msg,
                               structured_model=RoleListForm)

        if response.metadata is not None:
            json_form = response.metadata
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


def generate_role_data(num_time, index):
    json_out = asyncio.run(generate_role_agent(num_time))
    dict_save_to_json(json_out, '../data/roles_10000/role_list_en_{}.json'.format(index))

if __name__ == '__main__':
    total_num = 10000
    Path('../data/roles_{}'.format(total_num)).mkdir(parents=True, exist_ok=True)
    per_time = 10
    epoch_total_num = int(total_num / per_time)
    for epoch in range(per_time):
        process_s = []
        base_unm = int(epoch_total_num / per_time)
        base_i = epoch * base_unm
        for i in range(base_unm):
            sub_process = multiprocessing.Process(target=generate_role_data, args=(per_time, base_i + i + 1,))
            process_s.append(sub_process)
        for sub_process in process_s:
            sub_process.start()
        for sub_process in process_s:
            sub_process.join()
        print('All processes completed.')

    role_i = 0
    json_total = {'roles': []}
    folder = Path('../data/roles_{}'.format(total_num))
    for f in folder.iterdir():
        file_path = '../data/roles_{]/'.format(total_num) + f.name
        json_dict = json_read_to_dict(file_path)
        for i in range(len(json_dict['roles'])):
            json_dict['roles'][i]['name_id'] = role_i + 1
            json_total['roles'].append(json_dict['roles'][i])
            role_i += 1
        print(file_path)
    dict_save_to_json(json_total, '../data/role_list_en_{}.json'.format(total_num))
