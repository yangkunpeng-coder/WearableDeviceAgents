from typing import Literal
from pydantic import BaseModel, Field
from brain_s.deepseek_v4_pro import DeepseekV4ProBrain
from base_class.agent_base import AgentBase
from utils.file_tool import json_read_to_dict, dict_save_to_json

def evaluate_quality_prompt(form, question, answer):
    prompt = f"""
You are an expert evaluator of health decision support systems.base the "<Final Answer> </Final Answer>" for "# Question",
Note:1、The dimensions and rules of scoring are "#Scoring dimensions and scores";
2、“The output format "Format of the generated result" must be strictly followed, and the output should be in JSON.”

# Question
{question}

<Final Answer>
{answer}
</Final Answer>

# Scoring dimensions and scores
## Trustworthiness(1~5):
1 = no evidence
2 = vague evidence
3 = one supporting health indicator
4 = multiple supporting indicators
5 = multiple indicators with explicit reasoning linkage
## Transparency(1~5):
1 = conclusion only
2 = simple explanation
3 = partial reasoning chain
4 = complete reasoning chain
5 = complete reasoning chain with evidence sources
## Actionability(1~5):
1 = no recommendation
2 = vague recommendation
3 = general recommendation
4 = specific recommendation
5 = personalized and executable recommendation

# Format of the generated result
{form}
"""
    return prompt

class ScoreAnswerDatasForm(BaseModel):
    trustworthiness: Literal[1, 2, 3, 4, 5] = Field(description="trustworthiness score")
    transparency: Literal[1, 2, 3, 4, 5] = Field(description="transparency score")
    actionability: Literal[1, 2, 3, 4, 5] = Field(description="actionability score")

class ScoreLLM(AgentBase):
    def __init__(self, brain):
        super().__init__(brain)

    def run_agent(self, **kwargs):
        self.internal_inout["communications"] = []
        """
        Execute the agent task.
        Keyword Args:
            question (str):
            answer (dict): 
        """
        question = kwargs.get("question")
        if question is None:
            raise TypeError(
                "run_agent() missing required keyword-only argument 'question'"
            )
        answer = kwargs.get("answer")
        if answer is None:
            raise TypeError(
                "run_agent() missing required keyword-only argument 'answer'"
            )
        tokens_used = 0

        # agent_s analyse
        prompt = evaluate_quality_prompt(ScoreAnswerDatasForm.model_json_schema(), question, answer)
        sys_prompt = "You are an expert evaluator of health decision support systems."
        output, total_tokens = self.brain(sys_prompt=sys_prompt, user_prompt=prompt, formating=ScoreAnswerDatasForm)
        tokens_used += total_tokens
        output_dict = {"trustworthiness": output.trustworthiness,
                       "transparency": output.transparency,
                       "actionability": output.actionability}
        return output_dict, tokens_used


def get_evidence_string(data, max_width=10000, is_show_response=False):
    lines = ['# The data evidence']
    answer_dict_s = data['answer_dict_s']
    agent_tree_dict = data['agent_tree_dict']
    for answer_dict in answer_dict_s:
        if answer_dict['intent_type'] == 'query':
            lines.append(answer_dict['task_s'][0]['response'])
    lines.append('# The reasoning process')
    lines.append('The user question ' + agent_tree_dict['user_question'] + '.Decompose into the following multiple intents:')
    for intent_dict in agent_tree_dict['intent_s']:
        task_s = ''
        for task in intent_dict['task_s']:
            agent_name = task['agent_name']
            task_s += agent_name
            task_s += ' '
        lines.append(intent_dict['intent'] + ' implemented by {}'.format(task_s))
    return "\n".join(lines)

if __name__ == '__main__':
    deepseek_v4_pro_brain = DeepseekV4ProBrain()
    score_llm = ScoreLLM(deepseek_v4_pro_brain)
    intent_answer_s = json_read_to_dict('./mult_agent_answer_s.json')
    score_s = json_read_to_dict('./mult_agent_scores.json')
    last_role_id = score_s[-1]['role_id']
    for intent_answer in intent_answer_s[last_role_id: ]:
        role_id = intent_answer["role_id"]
        questions = intent_answer['question']
        mult_agents_answers = intent_answer['mult_agents_answer']
        llm_baseline_answers = intent_answer['llm_baseline_answer']
        role_score = {
            'role_id': role_id,
            'score_dict_s': []
        }
        for i in range(len(questions)):
            question = questions[i]
            mult_agents_answer = mult_agents_answers[i]
            llm_baseline_answer = llm_baseline_answers[i]
            mult_evidence_path = './mult_agent_output/role_{}_question_{}.json'.format(role_id, i+1)
            mult_agent_evidence = json_read_to_dict(mult_evidence_path)
            evidence = get_evidence_string(mult_agent_evidence)
            mult_agents_answer = evidence + '\n' + mult_agents_answer
            mult_agent_score, agent_tokens_used = score_llm.run_agent(question=question, answer=mult_agents_answer)
            llm_baseline_score, llm_tokens_used = score_llm.run_agent(question=question, answer=llm_baseline_answer)
            role_score['score_dict_s'].append({
                "question": question,
                "mult_agents_answer": mult_agents_answer,
                "llm_baseline_answer": llm_baseline_answer,
                "mult_agent_score": mult_agent_score,
                "llm_baseline_score": llm_baseline_score
            })
        score_s.append(role_score)
        dict_save_to_json(score_s, "./mult_agent_scores.json")
    dict_save_to_json(score_s, "./mult_agent_scores.json")
    print(score_s)
    