from abc import ABC, abstractmethod
from PyQt6.QtWidgets import QWidget

class FeatureModule(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @property
    @abstractmethod
    def icon(self) -> str:
        pass
        
    @property
    def is_standalone(self) -> bool:
        return False
        
    @abstractmethod
    def build_widget(self) -> QWidget:
        pass
        
    @abstractmethod
    def on_gesture(self, event_data: dict):
        pass
