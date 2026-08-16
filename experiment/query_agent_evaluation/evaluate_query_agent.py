import random
from tqdm import tqdm
from agent_s.query_agent import QueryAgent
from brain_s.deepseek_v4_pro import DeepseekV4ProBrain
from experiment.query_agent_evaluation.generate_query_question_agent import (random_date_type,
                                                                             create_query_question,
                                                                             expand_date_range,
                                                                             query_data_item,
                                                                             ReWriteAgent)
from experiment.query_agent_evaluation.llm_query_baseline import LLMQueryBaseline
from utils.file_tool import json_read_to_dict, dict_save_to_json

if __name__ == '__main__':
    # query_answer_dict_s = []
    deepseek_v4_pro_brain = DeepseekV4ProBrain()
    query_agent = QueryAgent(deepseek_v4_pro_brain)
    rewrite_agent = ReWriteAgent(deepseek_v4_pro_brain)
    llm_baseline = LLMQueryBaseline()
    query_answer_dict_s = json_read_to_dict('query_answer_dict_s.json')
    last_role_id = query_answer_dict_s[-1]['role_id']
    for role_i in tqdm(range(last_role_id + 1, 301, 1)):
        role_data_file_path = '../../data/roles_data_en_10000/role_data_{}.json'.format(role_i)
        file_dict = json_read_to_dict(role_data_file_path)
        answer_dict = {'role_id': file_dict['role']['name_id'],
                       'question_s': [],
                       'llm_baseline_answers' : [],
                       'llm_baseline_token_num': [],
                       'agent_answers': [],
                       'agent_token_num': [],
                       'true_answers': []}
        datetime_str_s = []
        for data_dict  in file_dict['data_s']:
            current_datetime = data_dict['datetime']
            datetime_str_s.append(current_datetime)
        query_question_ch = create_query_question()
        question_s = random.choices(query_question_ch, k=5)
        for question in question_s:
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
            task = question['task'].format(datetime_str)
            rewrite_task, tokens_used = rewrite_agent.run_agent(content=task, base_info=file_dict['role']['demographic_information'])
            data_type = question['data_type']
            true_result = query_data_item(file_dict, data_type, datetime_s)
            # agent_answer, agent_total_token = query_agent.query(rewrite_task, file_dict)
            # llm_baseline_answer, llm_baseline_total_token = llm_baseline.llm_query_baseline_fun(rewrite_task, file_dict)
            print(rewrite_task)
            print(true_result)
            agent_answer, agent_total_token = query_agent.run_agent(question = rewrite_task, role_data_dict = file_dict)
            llm_baseline_answer, llm_baseline_total_token = llm_baseline.run_llm_query_baseline(rewrite_task, file_dict)
            answer_dict['question_s'].append(rewrite_task)
            answer_dict['agent_answers'].append(agent_answer)
            answer_dict['agent_token_num'].append(agent_total_token)
            answer_dict['llm_baseline_answers'].append(llm_baseline_answer)
            answer_dict['llm_baseline_token_num'].append(llm_baseline_total_token)
            answer_dict['true_answers'].append(true_result)
        query_answer_dict_s.append(answer_dict)
        dict_save_to_json(query_answer_dict_s, 'query_answer_dict_s.json')
    # write answer
    dict_save_to_json(query_answer_dict_s, 'query_answer_dict_s.json')
    print(query_answer_dict_s)
