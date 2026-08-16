import numpy as np
from utils.file_tool import json_read_to_dict

if __name__ == '__main__':
    mult_agent_scores = json_read_to_dict('./mult_dim_trend_scores.json')
    mult_agent_trustworthiness_s, llm_baseline_trustworthiness_s = [], []
    mult_agent_transparency_s, llm_baseline_transparency_s = [], []
    mult_agent_actionability_s, llm_baseline_actionability_s = [], []
    for mult_agent_score in mult_agent_scores:
        score_dict_s = mult_agent_score['score_dict_s']
        for score_dict in score_dict_s:
            mult_agent_trustworthiness_s.append(score_dict['mult_agent_score']['trustworthiness'])
            mult_agent_transparency_s.append(score_dict['mult_agent_score']['transparency'])
            mult_agent_actionability_s.append(score_dict['mult_agent_score']['actionability'])
            llm_baseline_trustworthiness_s.append(score_dict['llm_baseline_score']['trustworthiness'])
            llm_baseline_transparency_s.append(score_dict['llm_baseline_score']['transparency'])
            llm_baseline_actionability_s.append(score_dict['llm_baseline_score']['actionability'])
    print('{},{},{}\n{},{},{}'.format(np.mean(mult_agent_trustworthiness_s), np.mean(mult_agent_transparency_s), np.mean(mult_agent_actionability_s),
                                      np.mean(llm_baseline_trustworthiness_s), np.mean(llm_baseline_transparency_s), np.mean(llm_baseline_actionability_s)))
