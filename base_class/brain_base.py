from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional, Type


class BrainBase(ABC):
    @abstractmethod
    def __call__(
        self,
        sys_prompt: str,
        user_prompt: str,
        formating: Optional[Type[BaseModel] | None] = None,
        **kwargs,
    ):
        pass