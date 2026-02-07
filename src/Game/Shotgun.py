from abc import ABC, abstractmethod
from Game.Bullet import Bullet
import numpy as np

class Shotgun(ABC):

    def __init__(self,):
        self.bullets_count = np.array([0,0])
        self.n_bullet = 0
        self.n_bullets = 0
    
    @abstractmethod
    def set_damage(self):
        pass
    
    @abstractmethod
    def shoot(self):
        pass
    
    def get_bullets_count(self):
        return self.bullets_count
    
    def get_n_bullet(self):
        return self.n_bullet
    
    def get_n_bullets(self):
        return self.n_bullets
    
    def is_empty(self):
        return self.n_bullets == 0
