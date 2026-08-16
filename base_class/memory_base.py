from abc import ABC, abstractmethod

class MemoryBase(ABC):
    def __init__(self, memory_path: str):
        self.memory_path = memory_path

    @abstractmethod
    def add_memory(self, data: dict):
        pass

