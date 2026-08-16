import json
from dataclasses import asdict
from brain_s.deepseek_v4_pro import DeepseekV4ProBrain
from agent_s.manager_agent import ManagerAgent
from llm_answer_baseline import LlmBaselineManagerAgent
from utils.file_tool import json_read_to_dict, dict_save_to_json

if __name__ == '__main__':
    question_s = json_read_to_dict('./mult_dim_trend.json')
    question_s = question_s['questions']
    deepseek_v4_pro_brain = DeepseekV4ProBrain()
    m_agent = ManagerAgent(deepseek_v4_pro_brain)
    llm_baseline = LlmBaselineManagerAgent()
    mult_agent_answer_s = json_read_to_dict('./mult_dim_trend_answer_s.json')
    last_role_id = mult_agent_answer_s[-1]['role_id']
    for role_id in range(last_role_id + 1, 11, 1):
        mult_agent_answer_dict = {
            "role_id": role_id,
            "question": question_s,
            "mult_agents_answer": [],
            "mult_agents_tokens_used": [],
            "mult_agents_intents": [],
            "llm_baseline_answer": [],
            "llm_baseline_tokens_used": [],
        }  # type: ignore
        role_data_path = '../../data/roles_data_en_10000/role_data_{}.json'.format(role_id)
        with open(role_data_path, 'r', encoding='utf-8') as f:
            role_data = json.load(f)
        for q_i, question in enumerate(question_s):
            m_agent_r = m_agent.run_agent(role_data_dict=role_data, content=question)
            with open("./mult_dim_trend/role_{}_question_{}.json".format(role_id, q_i+1), "w", encoding="utf-8") as f:
                json.dump(asdict(m_agent_r), f, indent=2, ensure_ascii=False)
            co_output, tokens_used, answer_dict_s = llm_baseline.run_llm_baseline_manager_agent(role_data, question)
            intents_str = ''
            for answer_dict in m_agent_r.answer_dict_s:
                intents_str += answer_dict['intent_type']
                intents_str += ', '
            intents_str = intents_str[:-2]
            mult_agent_answer_dict['mult_agents_answer'].append(m_agent_r.co_output)
            mult_agent_answer_dict['mult_agents_tokens_used'].append(m_agent_r.tokens_used)
            mult_agent_answer_dict['mult_agents_intents'].append(intents_str)
            mult_agent_answer_dict['llm_baseline_answer'].append(co_output)
            mult_agent_answer_dict['llm_baseline_tokens_used'].append(tokens_used)
        mult_agent_answer_s.append(mult_agent_answer_dict)
        dict_save_to_json(mult_agent_answer_s, 'mult_dim_trend_answer_s.json')
    # write answer
    dict_save_to_json(mult_agent_answer_s, "mult_dim_trend_answer_s.json")
