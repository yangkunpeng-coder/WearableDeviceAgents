import json
from textwrap import dedent
from typing import Literal, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field
from agent_s.advice_agent import AdviceAgent
from agent_s.analyse_agents import AnalyseAgent
from agent_s.query_agent import QueryAgent
from base_class.agent_base import AgentBase
from brain_s.deepseek_v4_pro import DeepseekV4ProBrain

def achieve_task(agent_name, agent_name_dict, agent_task, role_data_dict, **kwargs):
    base_info = ''
    response_str = ''
    token_num = 0
    if agent_name == 'query_agent':
        response_str, token_num = agent_name_dict['query_agent'].run_agent(question=agent_task, role_data_dict=role_data_dict, **kwargs)
    elif agent_name == 'advice_agent':
        response_str, token_num = agent_name_dict['advice_agent'].run_agent(question=agent_task, base_info=base_info, **kwargs)
    elif agent_name == 'analyse_agent':
        response_str, token_num = agent_name_dict['analyse_agent'].run_agent(question=agent_task, base_info=base_info, **kwargs)
    else:
        print('agent_s name error')
    return response_str, token_num

agent_s = {'query_agent': "Responsible for querying the user's exercise records, health records, and daily activity records.",
           'advice_agent': "Responsible for providing users with advice on maintaining health. If the user's lifestyle is unhealthy, it needs to point it out and prompt the user on how to improve.",
           'analyse_agent': "Responsible for analyzing user data and providing accurate analysis conclusions."}

def intent_prompt(form, content):
    prompt = f"""
    Analyze all the independent intents contained in "question", and extract the content corresponding to each independent intent, with the independent intents arranged in the order of the "question".
    Note:1、Output must be in JSON format without any explanation or additional text;2、answer in English;
    3、If a temporal constraint, population constraint, device constraint, or any other modifier appears before a list of coordinated items, assume the modifier applies to ALL items unless explicitly overridden.
    # question
    {content}
    # the meaning corresponding to the intent type
    1. query: refers to the user querying information, such as exercise records, health records, personal information;
    2. advice: refers to the user needing advice, such as exercise prescription, dietary prescription, health recommendations;
    3. analyse: refers to the need to analyze user data;
    # Format of the generated result
    {form}
    """
    prompt = dedent(prompt)
    return prompt

class IntentForm(BaseModel):
    intent_type: Literal['query', 'advice', 'analyse'] = Field(description="user's intent type")
    intent_context: str = Field(description="The content corresponding to the independent  intent type, accurately extracted from the user's question.")

class IntentsForm(BaseModel):
    intent_s: list[IntentForm] = Field(description="list of user's intents")

def task_prompt(form, intent_type, intent_context):
    agents_str = ''
    for k in agent_s.keys():
        agent_str = 'The responsibility of Agent {} is {}\n'.format(k, agent_s[k])
        agents_str += agent_str
    prompt = f"""
    It is necessary to assign one or more agents according to the user's intent. If multiple agents are assigned, they need to fulfill the user's intent in a sequential order. You do not need to implement it yourself; you only need to correctly arrange the agents.
    Note:1、Output must be in JSON format without any explanation or additional text;2、answer in English;
    # The responsibilities corresponding to different agents are
    {agents_str}
    # User's intent types and their corresponding descriptions
    The intent type is {intent_type}, The corresponding description is {intent_context}
    # Format of the generated result
    {form}
    """
    prompt = dedent(prompt)
    return prompt

class TaskForm(BaseModel):
    task_agent_name: Literal['query_agent',
                    'advice_agent',
                    'analyse_agent'] = Field(description="agent_s name")
    task_description: str = Field(description='A detailed description of the tasks arranged to be completed by the agent_s.')


class TasksForm(BaseModel):
    task_s: list[TaskForm] = Field(description='agent_s list')

def aggregate_prompt(question, answer_dict_s):
    answer_summary = '# question:\n{}\n'.format(question)
    for i, answer_dict in enumerate(answer_dict_s):
        answer_summary += '## sub question{}:\n{}\n'.format(i + 1, answer_dict['intent'])
        answer_summary += '## sub answer{}:\n{}\n'.format(i + 1, answer_dict['answer'])
    prompt = f"""
    Based on all the content of "##sub question" and "##sub answer", answer "#question:{question}", requiring accurate, complete, and logical responses.
    Note:1、Output must be without any explanation or additional text;2、answer in English;3、Please answer in Markdown format, using # for main sections, ## for subheadings, - for list items, and 1. for steps.
    4、title is #Answer
    {answer_summary}
    """
    prompt = dedent(prompt)
    return prompt

def summary_answer_dict_s(content, answer_dict_s):
    # summary
    answer_summary = '# question:\n{}\n'.format(content)
    for i, answer_dict in enumerate(answer_dict_s):
        answer_summary += '## {} -- sub question{}:\n{}\n'.format(answer_dict['intent_type'], i + 1,
                                                                  answer_dict['intent'])
        answer_summary += '## sub answer{}:\n{}\n'.format(i + 1, answer_dict['answer'])
        answer_summary += '## tasks{}\n'.format(i + 1)
        for ti, task_dict in enumerate(answer_dict['task_s']):
            agent_name = task_dict['agent_name']
            task_description = task_dict['task_description']
            task_response = task_dict['response']
            answer_summary += '{}、{}:{}\n{}\n'.format(ti + 1, agent_name, task_description, task_response)
    return answer_summary

@dataclass
class ManagerAgentReturn:
    co_output: str
    tokens_used: int
    answer_dict_s: list[dict]
    agent_tree_dict: dict

class ManagerAgent(AgentBase):
    # model
    def __init__(self, brain):
        super().__init__(brain)

    def run_agent(self, **kwargs):
        """
        Execute the agent task.
        Keyword Args:
            content (str): User content, required.
            role_data_dict (dict): Role data dictionary, defaults to {}.
        """
        self.internal_inout['communications'] = []
        content = kwargs.get("content")
        if content is None:
            raise TypeError("run_agent() missing required keyword-only argument 'content'")
        role_data_dict = kwargs.get("role_data_dict")
        if role_data_dict is None:
            raise TypeError("run_agent() missing required keyword-only argument 'role_data_dict'")

        tokens_used = 0
        # role_info = role_data_dict['role']['demographic_information']
        agent_name_dict = {
            'query_agent': QueryAgent(brain=self.brain),
            'advice_agent': AdviceAgent(brain=self.brain),
            'analyse_agent': AnalyseAgent(brain=self.brain),
        }

        de_intent_prompt = intent_prompt(form=IntentsForm.model_json_schema(), content=content)

        # de intent
        sys_prompt = "You are an intent recognition agent_s, responsible for recognizing the intent of smartwatch users."
        output_dict, total_tokens = self.brain(sys_prompt=sys_prompt,
                   user_prompt=de_intent_prompt, formating=IntentsForm)
        tokens_used += total_tokens
        intent_s = output_dict.intent_s
        self.internal_inout['communications'].append({'communication_name': 'de intent',
                                                    'sys_prompt': sys_prompt,
                                                    'user_prompt': de_intent_prompt,
                                                    'output_dict': output_dict.model_dump_json()})

        #de task
        # For each intent, assign tasks to the corresponding agents and decide the order in which to execute them.
        answer_dict_s = []
        agent_tree_dict: dict[str, Any] = {
            "agent_name": type(self).__name__,
            "user_question": content,
            "intent_s": [],
        }
        
        for intent_dict in intent_s:
            intent_type = intent_dict.intent_type
            intent_context = intent_dict.intent_context
            answer_dict = {
                "intent": intent_context,
                "intent_type": intent_type,
                "task_s": [],
            }
            de_task_prompt = task_prompt(TasksForm.model_json_schema(), intent_type, intent_context)

            sys_prompt = "You are a task assignment agent_s, responsible for correctly arranging agents based on the intent and intent description of the smartwatch user."
            output_dict, total_tokens = self.brain(
                sys_prompt=sys_prompt,
                user_prompt=de_task_prompt,
                formating=TasksForm,
            )
            tokens_used += total_tokens
            task_s = output_dict.task_s
            self.internal_inout["communications"].append(
                {
                    "communication_name": "de task",
                    "sys_prompt": sys_prompt,
                    "user_prompt": de_task_prompt,
                    "output_dict": output_dict.model_dump_json(),
                }
            )

            # achieve task
            result_str = ''
            for task_i, task_dict in enumerate(task_s):
                agent_name = task_dict.task_agent_name
                agent_task = task_dict.task_description
                add_info_dict = {'title': 'Additional reference information', 'additional_info': result_str}
                if task_i == len(task_s) - 1:
                    result_str, token_num = achieve_task(agent_name, agent_name_dict, agent_task, role_data_dict,
                                                    additional_info=add_info_dict)
                elif task_i == 0:
                    result_str, token_num = achieve_task(agent_name, agent_name_dict, agent_task, role_data_dict)
                else:
                    result_str, token_num =  achieve_task(agent_name, agent_name_dict, agent_task, role_data_dict,
                                                    additional_info=add_info_dict)
                tokens_used += token_num
                answer_dict['task_s'].append({'agent_name': agent_name,
                                              'task_description': agent_task,
                                              'response': result_str})
            answer_dict['answer'] = result_str
            answer_dict_s.append(answer_dict)
            agent_tree_dict['intent_s'].append(answer_dict)

        # aggregate
        co_prompt = aggregate_prompt(question=content, answer_dict_s=answer_dict_s)
        sys_prompt = "You are an aggregate agent_s, responsible for answering the question of smartwatch users."
        output_dict, total_tokens = self.brain(
            sys_prompt=sys_prompt,
            user_prompt=co_prompt,
        )
        co_output = output_dict
        tokens_used += total_tokens
        self.internal_inout["communications"].append(
            {
                "communication_name": "aggregate task",
                "sys_prompt": sys_prompt,
                "user_prompt": co_prompt,
                "output_dict": output_dict,
            }
        )
        r = ManagerAgentReturn(co_output, tokens_used, answer_dict_s, agent_tree_dict)
        return r


def test_manager_agent():
    # "Based on my data, what were my total steps and sleep scores on March 1 and March 15, and can you analyze how my exercise type on those days influenced my sleep? Finally, advise me on how to improve my sleep quality based on this analysis."
    content = "What was my sleep score, total steps, and the type of exercise I did on March 15, 2026?"
    role_data_path = '../data/roles_data_en_10000/role_data_1.json'
    with open(role_data_path, 'r', encoding='utf-8') as f:
        role_data = json.load(f)
    deepseek_v4_pro_brain = DeepseekV4ProBrain()
    m_agent = ManagerAgent(deepseek_v4_pro_brain)
    m_agent_r = m_agent.run_agent(role_data_dict=role_data, content=content)

    answer_dict_s_str = summary_answer_dict_s(content, m_agent_r.answer_dict_s)
    print(m_agent_r.agent_tree_dict)
    """
    print('question:\n{}'.format(content))
    print('tokens_used:\n{}'.format(m_agent_r.tokens_used))
    print('co_output:\n{}'.format(m_agent_r.co_output))
    print('answer_dict_s_str\n{}'.format(m_agent_r.answer_dict_s_str))
    """


if __name__ == '__main__':
    test_manager_agent()
