from Game.Bullet import Bullet
from Game.Shotgun import Shotgun
import numpy as np

class FakeShotgun(Shotgun):

    def __init__(self,chamber=[]):
        super().__init__()
        if chamber != []:
            self.chamber = chamber
            self.n_bullets = self.set_bullet_count()
        else:
            self.chamber = [Bullet(0)]*8
        self.current_bullet = self.chamber[self.n_bullet]
    
    def set_bullet_count(self):
        n_bullets = 0
        for bullet in self.chamber:
            if bullet.get_type() == 2:
                self.bullets_count[0] += 1
            elif bullet.get_type() == 3:
                self.bullets_count[1] += 1
            n_bullets += 1
        self.chamber += [Bullet()]*(8-len(self.chamber))
        if self.bullets_count[0] == 0:
            if self.get_n_bullets() == 8:
                self.invert_bullet_pos(-1)
            else:
                if n_bullets < len(self.chamber):
                    self.chamber[n_bullets] = Bullet(2)
                    self.bullets_count[0] += 1
                    n_bullets += 1
        elif self.bullets_count[1] == 0:
            if self.get_n_bullets() == 8:
                self.invert_bullet_pos(-1)
            else:
                if n_bullets < len(self.chamber):
                    self.chamber[n_bullets] = Bullet(3)
                    self.bullets_count[1] += 1
                    n_bullets += 1
        return n_bullets

    def get_current_bullet(self):
        return self.current_bullet
    
    def set_damage(self):
        self.current_bullet.damage = self.current_bullet.damage*2
    
    def invert_bullet_pos(self,pos):
        if self.chamber[pos].type == 2:
            self.chamber[pos].set_type(3)
            self.bullets_count[0] -= 1
            self.bullets_count[1] += 1
        elif self.chamber[pos].type == 3:
            self.chamber[pos].set_type(2)
            self.bullets_count[1] -= 1
            self.bullets_count[0] += 1
    
    def invert_bullet(self):
        self.invert_bullet_pos(self.n_bullet)
    
    def shoot(self):
        current_damage = self.current_bullet.get_damage()
        if self.current_bullet.get_type() == 2:
            self.bullets_count[0] -= 1
        elif self.current_bullet.get_type() == 3:
            self.bullets_count[1] -= 1
        # decrement remaining bullets count
        if self.n_bullets > 0:
            self.n_bullets -= 1

        # mark the current chamber slot as used if index is valid
        if 0 <= self.n_bullet < len(self.chamber):
            self.chamber[self.n_bullet] = Bullet(1)
        else:
            # nothing to mark, shotgun is effectively empty
            self.empty = True

        # advance to next bullet index
        self.n_bullet += 1
        if self.n_bullet >= len(self.chamber):
            self.empty = True
            # keep current_bullet as a safe sentinel Bullet(0)
            self.current_bullet = Bullet(0)
        else:
            self.current_bullet = self.chamber[self.n_bullet]
        return current_damage
    
    def __str__(self):
        return str(self.chamber)
    
    def __repr__(self):
        return str(self)
        