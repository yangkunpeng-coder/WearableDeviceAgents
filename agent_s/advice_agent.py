from textwrap import dedent
from pydantic import BaseModel, Field
from base_class.agent_base import AgentBase

def advice_prompt(form, question, base_info, **kwargs):
    additional_info = ''
    for key, value in kwargs.items():
        additional_info += '###{}\n'.format(value['title'])
        additional_info += '{}\n'.format(value['additional_info'])
    prompt = f"""
    based on your "basic information" and "additional_info", answer the "question".
    Note:1、Output must be in JSON format without any explanation or additional text;2、answer in English;
    {additional_info}
    # question
    {question}
    # user's basic information
    {base_info}
    ### Format of the generated result
    {form}
    """
    prompt = dedent(prompt)
    return prompt

class AdviceForm(BaseModel):
    advice: list[str] = Field(description="list of health recommendations for the user")

class AdviceAgent(AgentBase):
    def __init__(self, brain):
        super().__init__(brain)
    
    def run_agent(self, **kwargs):
        """
        Execute the agent task.
        Keyword Args:
            question (str): User question, required.
            base_info (dict): base_info data dictionary, defaults to {}.
        """
        self.internal_inout["communications"] = []
        question = kwargs.get("question")
        if question is None:
            raise TypeError("run_agent() missing required keyword-only argument 'question'")
        base_info = kwargs.get("base_info")
        if base_info is None:
            raise TypeError("run_agent() missing required keyword-only argument 'base_info'")

        tokens_used = 0
        if "question" in kwargs:
            del kwargs["question"]
        if "base_info" in kwargs:
            del kwargs["base_info"]
        prompt = advice_prompt(form=AdviceForm.model_json_schema(), question=question, base_info=base_info,
                                     **kwargs)
        sys_prompt = "You are a health advice agent_s, responsible for providing users with health-related advice."
        output_dict, total_tokens = self.brain(sys_prompt=sys_prompt,
                   user_prompt=prompt, formating=AdviceForm)

        tokens_used += total_tokens
        health_advices = output_dict.advice

        advice_str = ''
        for i, health_advice in enumerate(health_advices):
            advice_str += '{}、{}\n'.format(i + 1, health_advice)

        self.internal_inout['communications'].append({'communication_name': 'advice_inout',
                                               'sys_prompt': sys_prompt,
                                               'user_prompt': prompt,
                                               'output_dict': output_dict.model_dump_json()})

        return advice_str, tokens_used

if __name__ == '__main__':
    ...
