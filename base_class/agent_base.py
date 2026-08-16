from abc import ABC, abstractmethod
from typing import Any
from base_class.brain_base import BrainBase


class AgentBase(ABC):
    def __init__(self, brain: BrainBase):
        self.brain = brain
        self.internal_inout: dict[str, Any]= {'agent_name': type(self).__name__,
                                'communications': []}

    @abstractmethod
    def run_agent(self, **kwargs):
        pass

    def get_internal_brain_inout(self, **kwargs):
        return self.internal_inout
