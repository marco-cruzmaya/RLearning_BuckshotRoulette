import copy

class Bullet:
    def __init__(self,type=0):
        self.type = type
        self.damage = self.set_damage()
    
    def set_damage(self):
        damage = {0:0, 1:0, 2:-1, 3:0}
        self.damage = damage[self.type]
        return self.damage
    
    def set_type(self,type):
        self.type = type
        self.damage = self.set_damage()
    
    def get_type(self):
        return self.type
    
    def get_damage(self):
        return self.damage
    
    def __str__(self):
        type_str = {0:'Uknown', 1:'Used', 2:'Life', 3:'Blank'}
        return type_str[self.type]
    
    def __repr__(self):
        return str(self)
    
    def copy(self):
        return copy.deepcopy(self)