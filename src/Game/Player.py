from abc import ABC, abstractmethod
from Game.Shotgun import Shotgun
from Game.Bullet import Bullet
from Game.Loadout import Loadout
import copy

class Player(ABC):

    def __init__(self,shotgun,loadout, health):
        self.health = health
        self.max_health = health
        self.loadout = loadout
        self.copy_loadout = copy.deepcopy(self.loadout)
        self.can_shoot = True
        self.can_take_action = True
        self.shotgun = shotgun
        self.chamber = [Bullet(0)]*8
        self.bullets_count = self.shotgun.get_bullets_count()
        self.n_bullet = self.shotgun.get_n_bullet()
    
    def shoot(self,player):
        if self.can_shoot:
            player.health += self.shotgun.shoot()
            self.n_bullet += 1
        return self.can_shoot
    
    def get_loadout(self):
        return self.loadout
    
    def get_bullets_count(self):
        return self.bullets_count
    
    def get_items(self):
        return self.loadout.items
    
    def get_chamber(self):
        return [bullet.get_type() for bullet in self.chamber]
    
    def set_can_take_action(self,flag):
        self.can_take_action = flag
    
    def is_alive(self):
        return self.health > 0
    
    def set_shotgun(self, shotgun):
        self.shotgun = shotgun
        self.n_bullet = self.shotgun.get_n_bullet()
        self.bullets_count = self.shotgun.get_bullets_count()
        self.chamber = [Bullet(0)]*8
    
    def get_current_bullet(self):
        return self.chamber[self.n_bullet]
    
    @abstractmethod
    def pick_item(self,pos):
        pass

    @abstractmethod
    def put_item(self,item):
        pass

    def __str__(self):
        return f"Health:{self.health}\nChamber:{self.chamber}\nBullets:{self.bullets_count}\nLoadout:{self.loadout}"
    
    def __repr__(self):
        return str(self)