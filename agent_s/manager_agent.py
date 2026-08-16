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


TASK_TYPE_TO_AGENT = {
    'query': 'query_agent',
    'analyse': 'analyse_agent',
    'advice': 'advice_agent',
}


TASK_TYPE_RESPONSIBILITIES = {
    'query': (
        "Retrieve the user's exercise records, health monitoring records, "
        "sleep records, or daily activity records required by the current intent."
    ),
    'analyse': (
        "Analyze and organize results produced by the explicitly dependent query tasks, "
        "including longitudinal comparison or multi-indicator analysis."
    ),
    'advice': (
        "Generate health-support recommendations based on the explicitly dependent "
        "query evidence and/or analysis results."
    ),
}


def achieve_task(agent_name, agent_name_dict, agent_task, role_data_dict, **kwargs):
    """Execute one task with the selected specialized agent."""
    base_info = ''
    response_str = ''
    token_num = 0

    if agent_name == 'query_agent':
        response_str, token_num = agent_name_dict['query_agent'].run_agent(
            question=agent_task,
            role_data_dict=role_data_dict,
            **kwargs,
        )
    elif agent_name == 'advice_agent':
        response_str, token_num = agent_name_dict['advice_agent'].run_agent(
            question=agent_task,
            base_info=base_info,
            **kwargs,
        )
    elif agent_name == 'analyse_agent':
        response_str, token_num = agent_name_dict['analyse_agent'].run_agent(
            question=agent_task,
            base_info=base_info,
            **kwargs,
        )
    else:
        raise ValueError(f'Unknown agent name: {agent_name}')

    return response_str, token_num


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
    return dedent(prompt)


class IntentForm(BaseModel):
    intent_type: Literal['query', 'advice', 'analyse'] = Field(description="user's intent type")
    intent_context: str = Field(description="The content corresponding to the independent intent type, accurately extracted from the user's question.")


class IntentsForm(BaseModel):
    intent_s: list[IntentForm] = Field(description="list of user's intents")


def task_prompt(form, intent_type, intent_context):
    prompt = f"""
    Decompose the given user intent into one or more executable tasks.

    Each task must have:
    1. a unique task_id;
    2. a task_type selected from query, analyse, or advice;
    3. a detailed task_description;
    4. a dependencies list containing the task_id values of the previous tasks whose results are required to execute the current task.

    Follow these rules:
    1. Output tasks in their execution order.
    2. Dependencies can only refer to tasks that appear earlier in the current task list.
    3. Do not create dependencies across different user intents.
    4. A query task retrieves information from the user's structured wearable records and therefore must have an empty dependencies list.
    5. An analyse task should depend on one or more previous query tasks when their retrieved results are required for the analysis.
    6. An advice task may depend on previous query tasks, analyse tasks, or both, depending on the evidence required to generate the recommendation.
    7. Only create tasks that are necessary to fulfill the current intent. Avoid redundant tasks.
    8. The task_type of the final task must be the same as the current intent_type.
    9. Preserve all temporal constraints, health indicators, and other necessary conditions from the intent_context in the corresponding task descriptions.
    10. Output must be valid JSON without any explanation or additional text.
    11. Answer in English.

    # Current intent
    intent_type:
    {intent_type}

    intent_context:
    {intent_context}

    # Responsibilities of different task types
    query:
    {TASK_TYPE_RESPONSIBILITIES['query']}

    analyse:
    {TASK_TYPE_RESPONSIBILITIES['analyse']}

    advice:
    {TASK_TYPE_RESPONSIBILITIES['advice']}

    # Format of the generated result
    {form}
    """
    return dedent(prompt)


class TaskForm(BaseModel):
    task_id: str = Field(description="Unique task identifier within the current intent, e.g. t1, t2.")
    task_type: Literal['query', 'analyse', 'advice'] = Field(description="Task type used to deterministically select the specialized agent.")
    task_description: str = Field(description="A detailed executable description of the task.")
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of previous tasks in the current intent whose results are required to execute this task.")


class TasksForm(BaseModel):
    task_s: list[TaskForm] = Field(description="Ordered executable task list")


def validate_task_sequence(task_s: list[TaskForm], intent_type: str):
    """
    Validate explicit intra-intent dependencies.

    The checks intentionally mirror the paper definition:
      * task_id is unique within one intent;
      * dependencies can only point to earlier tasks in the same intent;
      * query tasks have no dependencies;
      * analyse tasks must have at least one dependency and all of their dependencies
        must be query tasks;
      * the terminal task type equals the intent type.
    """
    if not task_s:
        raise ValueError('Task decomposition returned an empty task list.')

    previous_task_types: dict[str, str] = {}

    for task in task_s:
        task_id = task.task_id.strip()
        if not task_id:
            raise ValueError('task_id cannot be empty.')

        if task_id in previous_task_types:
            raise ValueError(f'Duplicate task_id within current intent: {task_id}')

        dependencies = task.dependencies
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(
                f'Task {task_id} contains duplicate dependencies: {dependencies}'
            )

        invalid_dependencies = [
            dep_id for dep_id in dependencies if dep_id not in previous_task_types
        ]
        if invalid_dependencies:
            raise ValueError(
                f'Task {task_id} has dependencies that are not previous tasks in '
                f'the current intent: {invalid_dependencies}'
            )

        if task.task_type == 'query' and dependencies:
            raise ValueError(
                f'Query task {task_id} must have an empty dependencies list.'
            )

        if task.task_type == 'analyse':
            if not dependencies:
                raise ValueError(
                    f'Analyse task {task_id} must depend on at least one Query Task.'
                )
            non_query_dependencies = [
                dep_id
                for dep_id in dependencies
                if previous_task_types[dep_id] != 'query'
            ]
            if non_query_dependencies:
                raise ValueError(
                    f'Analyse task {task_id} may only depend on Query Tasks; got '
                    f'{non_query_dependencies}.'
                )

        previous_task_types[task_id] = task.task_type

    if task_s[-1].task_type != intent_type:
        raise ValueError(
            'The final task type must match the current intent type: '
            f'intent={intent_type}, final_task={task_s[-1].task_type}'
        )


def collect_dependency_results(
    dependencies: list[str],
    local_result_store: dict[str, dict[str, Any]],
):
    """Collect only the results explicitly named by `dependencies`."""
    dependency_results = []

    for dep_id in dependencies:
        if dep_id not in local_result_store:
            raise KeyError(
                f'Dependency {dep_id} has not been executed in the current intent.'
            )
        dependency_results.append(local_result_store[dep_id])
    return dependency_results


def build_additional_info(dependency_results: list[dict[str, Any]]):
    """
    Convert explicitly dependent task results to the existing additional_info format
    expected by AnalyseAgent / AdviceAgent.
    """
    if not dependency_results:
        dependency_text = ''
    else:
        blocks = []
        for item in dependency_results:
            blocks.append(
                '\n'.join(
                    [
                        f"Task ID: {item['task_id']}",
                        f"Task Type: {item['task_type']}",
                        f"Task Description: {item['task_description']}",
                        'Task Result:',
                        str(item['response']),
                    ]
                )
            )
        dependency_text = '\n\n'.join(blocks)

    return {
        'title': 'Explicitly dependent task results',
        'additional_info': dependency_text,
    }


def aggregate_prompt(question, answer_dict_s):
    answer_summary = '# question:\n{}\n'.format(question)
    for i, answer_dict in enumerate(answer_dict_s):
        answer_summary += '## sub question{}:\n{}\n'.format(
            i + 1, answer_dict['intent']
        )
        answer_summary += '## sub answer{}:\n{}\n'.format(
            i + 1, answer_dict['answer']
        )

    prompt = f"""
    Based on all the content of "##sub question" and "##sub answer", answer "#question:{question}", requiring accurate, complete, and logical responses.
    Note:1、Output must be without any explanation or additional text;2、answer in English;3、Please answer in Markdown format, using # for main sections, ## for subheadings, - for list items, and 1. for steps.
    4、title is #Answer
    {answer_summary}
    """
    return dedent(prompt)


def summary_answer_dict_s(content, answer_dict_s):
    answer_summary = '# question:\n{}\n'.format(content)
    for i, answer_dict in enumerate(answer_dict_s):
        answer_summary += '## {} -- sub question{}:\n{}\n'.format(
            answer_dict['intent_type'], i + 1, answer_dict['intent']
        )
        answer_summary += '## sub answer{}:\n{}\n'.format(
            i + 1, answer_dict['answer']
        )
        answer_summary += '## tasks{}\n'.format(i + 1)

        for ti, task_dict in enumerate(answer_dict['task_s']):
            answer_summary += (
                '{}、{} [{}] {}\n'
                'dependencies={}\n'
                '{}\n'.format(
                    ti + 1,
                    task_dict['task_id'],
                    task_dict['task_type'],
                    task_dict['task_description'],
                    task_dict['dependencies'],
                    task_dict['response'],
                )
            )

    return answer_summary


@dataclass
class ManagerAgentReturn:
    co_output: str
    tokens_used: int
    answer_dict_s: list[dict]
    agent_tree_dict: dict


class ManagerAgent(AgentBase):
    def __init__(
        self,
        brain,
        dependency_mode: Literal['explicit', 'all_previous'] = 'explicit',
        agent_overrides: dict[str, AgentBase] | None = None,
    ):
        super().__init__(brain)
        if dependency_mode not in {'explicit', 'all_previous'}:
            raise ValueError(
                "dependency_mode must be either 'explicit' or 'all_previous'."
            )
        self.dependency_mode = dependency_mode
        self.agent_overrides = dict(agent_overrides or {})

    def run_agent(self, **kwargs):
        """
        Execute the manager-agent workflow.

        Keyword Args:
            content (str): User content, required.
            role_data_dict (dict): User wearable data, required.
        """
        self.internal_inout['communications'] = []

        content = kwargs.get('content')
        if content is None:
            raise TypeError(
                "run_agent() missing required keyword-only argument 'content'"
            )

        role_data_dict = kwargs.get('role_data_dict')
        if role_data_dict is None:
            raise TypeError(
                "run_agent() missing required keyword-only argument 'role_data_dict'"
            )

        tokens_used = 0

        agent_name_dict = {
            'query_agent': QueryAgent(brain=self.brain),
            'advice_agent': AdviceAgent(brain=self.brain),
            'analyse_agent': AnalyseAgent(brain=self.brain),
        }
        unknown_overrides = set(self.agent_overrides) - set(agent_name_dict)
        if unknown_overrides:
            raise ValueError(
                f'Unknown agent override names: {sorted(unknown_overrides)}'
            )
        agent_name_dict.update(self.agent_overrides)

        # -------------------------------------------------------------
        # 1. Multi-intent recognition
        # -------------------------------------------------------------
        de_intent_prompt = intent_prompt(
            form=IntentsForm.model_json_schema(),
            content=content,
        )

        sys_prompt = (
            'You are an intent recognition agent_s, responsible for recognizing '
            'the intent of smartwatch users.'
        )
        output_dict, total_tokens = self.brain(
            sys_prompt=sys_prompt,
            user_prompt=de_intent_prompt,
            formating=IntentsForm,
        )
        tokens_used += total_tokens
        intent_s = output_dict.intent_s

        self.internal_inout['communications'].append(
            {
                'communication_name': 'de intent',
                'sys_prompt': sys_prompt,
                'user_prompt': de_intent_prompt,
                'output_dict': output_dict.model_dump_json(),
            }
        )

        # -------------------------------------------------------------
        # 2. Intent-local task decomposition and dependency-aware execution
        # -------------------------------------------------------------
        answer_dict_s = []
        agent_tree_dict: dict[str, Any] = {
            'agent_name': type(self).__name__,
            'user_question': content,
            'dependency_mode': self.dependency_mode,
            'intent_s': [],
        }

        for intent_dict in intent_s:
            intent_type = intent_dict.intent_type
            intent_context = intent_dict.intent_context

            answer_dict = {
                'intent': intent_context,
                'intent_type': intent_type,
                'task_s': [],
            }

            de_task_prompt = task_prompt(
                TasksForm.model_json_schema(),
                intent_type,
                intent_context,
            )

            sys_prompt = (
                'You are a task decomposition agent responsible for decomposing a '
                "smartwatch user's intent into executable tasks and explicitly "
                'specifying dependencies among tasks.'
            )
            output_dict, total_tokens = self.brain(
                sys_prompt=sys_prompt,
                user_prompt=de_task_prompt,
                formating=TasksForm,
            )
            tokens_used += total_tokens
            task_s = output_dict.task_s

            # Fail explicitly if the generated task graph violates the paper definition.
            validate_task_sequence(task_s, intent_type)

            self.internal_inout['communications'].append(
                {
                    'communication_name': 'de task',
                    'sys_prompt': sys_prompt,
                    'user_prompt': de_task_prompt,
                    'output_dict': output_dict.model_dump_json(),
                }
            )

            # Each intent has its own local task-result store. This prevents
            # cross-intent dependencies by construction.
            local_result_store: dict[str, dict[str, Any]] = {}

            for task in task_s:
                task_id = task.task_id
                task_type = task.task_type
                agent_name = TASK_TYPE_TO_AGENT[task_type]
                agent_task = task.task_description
                dependencies = list(task.dependencies)

                if task_type == 'query':
                    # Query Task directly accesses the structured wearable records.
                    result_str, token_num = achieve_task(
                        agent_name,
                        agent_name_dict,
                        agent_task,
                        role_data_dict,
                    )
                else:
                    if self.dependency_mode == 'explicit':
                        # Full model: pass only the results named by the task graph.
                        effective_dependencies = dependencies
                    else:
                        # Ablation: reproduce the common implicit-history design by
                        # exposing every earlier result in the current intent.
                        effective_dependencies = list(local_result_store)

                    dependency_results = collect_dependency_results(
                        effective_dependencies,
                        local_result_store,
                    )
                    add_info_dict = build_additional_info(dependency_results)

                    result_str, token_num = achieve_task(
                        agent_name,
                        agent_name_dict,
                        agent_task,
                        role_data_dict,
                        additional_info=add_info_dict,
                    )

                tokens_used += token_num

                task_result = {
                    'task_id': task_id,
                    'task_type': task_type,
                    'agent_name': agent_name,
                    'task_description': agent_task,
                    'dependencies': dependencies,
                    'effective_dependencies': (
                        dependencies
                        if task_type == 'query'
                        else effective_dependencies
                    ),
                    'response': result_str,
                }

                # Save by task_id so later tasks can explicitly retrieve it.
                local_result_store[task_id] = task_result
                answer_dict['task_s'].append(task_result)

            # The final task is guaranteed to match intent_type by validation.
            terminal_task_id = task_s[-1].task_id
            answer_dict['answer'] = local_result_store[terminal_task_id]['response']

            answer_dict_s.append(answer_dict)
            agent_tree_dict['intent_s'].append(answer_dict)

        # -------------------------------------------------------------
        # 3. Intent-level aggregation only
        # -------------------------------------------------------------
        co_prompt = aggregate_prompt(
            question=content,
            answer_dict_s=answer_dict_s,
        )
        sys_prompt = (
            'You are an aggregate agent_s, responsible for answering the question '
            'of smartwatch users.'
        )
        output_dict, total_tokens = self.brain(
            sys_prompt=sys_prompt,
            user_prompt=co_prompt,
        )
        co_output = output_dict
        tokens_used += total_tokens

        self.internal_inout['communications'].append(
            {
                'communication_name': 'aggregate task',
                'sys_prompt': sys_prompt,
                'user_prompt': co_prompt,
                'output_dict': output_dict,
            }
        )

        return ManagerAgentReturn(
            co_output=co_output,
            tokens_used=tokens_used,
            answer_dict_s=answer_dict_s,
            agent_tree_dict=agent_tree_dict,
        )


def test_manager_agent():
    content = (
        'What were my atrial fibrillation events, exercise frequency, and sleep '
        'quality over the past three weeks, and do these changes indicate anything '
        'I should pay attention to?'
    )
    role_data_path = '../data/roles_data_en_10000/role_data_1.json'

    with open(role_data_path, 'r', encoding='utf-8') as f:
        role_data = json.load(f)

    deepseek_v4_pro_brain = DeepseekV4ProBrain()
    m_agent = ManagerAgent(deepseek_v4_pro_brain)
    m_agent_r = m_agent.run_agent(
        role_data_dict=role_data,
        content=content,
    )

    answer_dict_s_str = summary_answer_dict_s(
        content,
        m_agent_r.answer_dict_s,
    )

    print(json.dumps(m_agent_r.agent_tree_dict, ensure_ascii=False, indent=2))
    print('tokens_used:\n{}'.format(m_agent_r.tokens_used))
    print('co_output:\n{}'.format(m_agent_r.co_output))
    print('answer_dict_s_str:\n{}'.format(answer_dict_s_str))


if __name__ == '__main__':
    test_manager_agent()
