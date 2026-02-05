import copy
from abc import ABC, abstractmethod
class Loadout(ABC):

    def __init__(self):
        self.free_slots = [i for i in range(8)]
        self.items = [0]*8
    
    @abstractmethod
    def insert_item(self,item):
        pass
    
    @abstractmethod
    def take_item(self,pos):
        pass
    
    @abstractmethod
    def take_items(self,items):
        pass

    def __str__(self):
        return str(self.items)
    
    def __repr__(self):
        return str(self)
    
    def copy(self):
        return copy.deepcopy(self)