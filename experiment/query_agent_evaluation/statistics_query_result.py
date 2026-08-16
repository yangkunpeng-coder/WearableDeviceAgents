from utils.file_tool import json_read_to_dict

if __name__ == '__main__':
    query_answer_s = json_read_to_dict('query_answer_judge.json')
    judge_query_answer_s = []
    true_agent_num = 0
    true_llm_baseline_num = 0
    total_num = 0
    llm_baseline_token_num = 0
    agent_token_num = 0
    for query_answer in query_answer_s:
        judge_agents = query_answer['judge_agents']
        judge_llm_baseline = query_answer['judge_llm_baseline']
        llm_baseline_token_num_s = query_answer['llm_baseline_token_num']
        agent_token_num_s = query_answer['agent_token_num']
        for i in range(len(judge_agents)):
            if judge_agents[i] == 'True':
                true_agent_num += 1
            if judge_llm_baseline[i] == 'True':
                true_llm_baseline_num += 1
            llm_baseline_token_num += llm_baseline_token_num_s[i]
            agent_token_num += agent_token_num_s[i]
            total_num += 1
    print(true_agent_num / total_num)
    print(true_llm_baseline_num / total_num)
    print(agent_token_num / total_num)
    print(llm_baseline_token_num / total_num)
