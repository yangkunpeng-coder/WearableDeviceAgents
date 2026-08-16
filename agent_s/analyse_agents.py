from textwrap import dedent
from pydantic import BaseModel, Field
from base_class.agent_base import AgentBase

def analyse_data_prompt(form, question, base_info, **kwargs):
    additional_info = ''
    for key, value in kwargs.items():
        additional_info += '#{}\n'.format(value['title'])
        additional_info += '{}\n'.format(value['additional_info'])
    prompt = f"""
    Based on your "basic information" and "additional_info", answer the "question".
    Note:1、Output must be in JSON format without any explanation or additional text;2、answer in English;
    {additional_info}
    # question
    {question}
    # basic information
    {base_info}
    # Format of the generated result
    {form}
    """
    prompt = dedent(prompt)
    return prompt


class AnalyseDatasForm(BaseModel):
    analyse_data: list[str] = Field(description="list of data analysis results")


class AnalyseAgent(AgentBase):
    def __init__(self, brain):
        super().__init__(brain)

    def run_agent(self, question, base_info, **kwargs):
        """
        Execute the agent task.
        Keyword Args:
            question (str): User question, required.
            base_info (dict): base_info data dictionary, defaults to {}.
        """
        self.internal_inout["communications"] = []
        tokens_used = 0
        prompt = analyse_data_prompt(form=AnalyseDatasForm.model_json_schema(), question=question, base_info=base_info,
                                     **kwargs)
        sys_prompt = "You are a data analysis agent_s responsible for analyzing user data and providing reliable conclusions."
        output_dict, total_tokens = self.brain(sys_prompt=sys_prompt,
                   user_prompt=prompt, formating=AnalyseDatasForm)

        tokens_used += total_tokens
        analyse_data_s = output_dict.analyse_data
        datas_str = ''
        for i, analyse_data in enumerate(analyse_data_s):
            datas_str += '{}、{}\n'.format(i + 1, analyse_data)
            
        self.internal_inout['communications'].append({'communication_name':'analyse_inout',
                                                 'sys_prompt': sys_prompt,
                                                 'user_prompt': prompt,
                                                 'output_dict': output_dict.model_dump_json()})

        return datas_str, tokens_used


if __name__ == '__main__':
    ...
