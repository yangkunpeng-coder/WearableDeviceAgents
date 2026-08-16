import asyncio
from textwrap import dedent
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from typing import Literal
from pydantic import BaseModel, Field
from agent_s.agent_config import get_model
from utils.file_tool import json_read_to_dict, dict_save_to_excel, dict_save_to_json


def judge_answer_prompt(form, question, answer, true_answer, **kwargs):
    additional_info = ''
    for key, value in kwargs.items():
        additional_info += '###{}\n'.format(value['title'])
        additional_info += '{}\n'.format(value['additional_info'])
    prompt = f"""
    "user's answer" is the user's response to the "question", the correct answer is "true_answer", and it is required to determine whether "user's answer" is correct.
    Note:1 if user's answer is 'True', otherwise 'False'
    {additional_info}
    ### question
    {question}
    ### user's answer
    {answer}
    ### true_answer
    {true_answer}
    ### Format of the generated result
    {form}
    """
    prompt = dedent(prompt)
    return prompt


class JudgeAnswerDatasForm(BaseModel):
    judge_data: Literal['True', 'False'] = Field(description="1 if user's answer is 'True', otherwise 'False'.")


async def judge_answer_agent(question, answer, true_answer, model_type='deepseek', **kwargs):
    # model
    model, chat_form = get_model(model_type, 1.5)

    # tool
    toolkit = Toolkit()

    # agent_s
    agent = ReActAgent(
        name='judge_answer_agent',
        sys_prompt='You are a translate language agent_s responsible for rewrite user input',
        model=model,
        formatter=chat_form,
        toolkit=toolkit,
        memory=InMemoryMemory(),
        print_hint_msg=False,
    )

    prompt = judge_answer_prompt(form=JudgeAnswerDatasForm.model_json_schema(), question=question, answer=answer, true_answer=true_answer, **kwargs)
    msg = Msg("judge_answer_agent", prompt, "user")
    msg = await agent(msg, structured_model=JudgeAnswerDatasForm)
    judge_data = msg.metadata['judge_data']
    return judge_data

async def judge_tasks(question, agent_answer, llm_baseline_answer, true_answer):
    agent_judge, llm_baseline_judge = await asyncio.gather(
        judge_answer_agent(question, agent_answer, true_answer),
        judge_answer_agent(question, llm_baseline_answer, true_answer)
    )
    return agent_judge, llm_baseline_judge

if __name__ == '__main__':
    query_answer_s = json_read_to_dict('query_answer_dict_s.json')
    judge_query_answer_s = []
    for query_answer in query_answer_s:
        query_answer['judge_agents'] = []
        query_answer['judge_llm_baseline'] = []
        for i in range (len(query_answer['true_answers'])):
            question = query_answer['question_s'][i]
            agent_answer = query_answer['agent_answers'][i]
            llm_baseline_answer = query_answer['llm_baseline_answers'][i]
            true_answer = query_answer['true_answers'][i]
            agent_judge, llm_baseline_judge = asyncio.run(judge_tasks(question, agent_answer, llm_baseline_answer, true_answer))
            query_answer['judge_agents'].append(agent_judge)
            query_answer['judge_llm_baseline'].append(llm_baseline_judge)
        judge_query_answer_s.append(query_answer)
    dict_save_to_json(judge_query_answer_s, 'query_answer_judge.json')
