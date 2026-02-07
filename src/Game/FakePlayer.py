from Game.FakeShotgun import FakeShotgun
from Game.FakeLoadout import FakeLoadout
from Game.Bullet import Bullet
from Game.Player import Player
import random as rnd
import copy


class FakePlayer(Player):

    def __init__(self,health=0,shotgun=None,loadout=None):
        if shotgun is None:
            shotgun = FakeShotgun()
        if loadout is None:
            loadout = FakeLoadout()
        super().__init__(shotgun,loadout,health)
    
    def pick_item(self,pos):
        item = self.loadout.take_item(pos)
        return item
    
    def put_item(self,item):
        if self.loadout.insert_item(item):
            self.copy_loadout = self.loadout.copy()
            return True
        return False
        
    def put_items(self,items):
        for item in items:
            if self.put_item(item):
                continue
            else:
                return False
        self.copy_loadout = self.loadout.copy()
        return True
    
    def use_item(self, pos, o_player):
        if pos >= 8:
            return False
        match self.pick_item(pos):
            case 0: #None
                return False
            case 1: #Adrenaline
                if not self.can_shoot:
                    return False
                self.can_shoot = False
                self.copy_loadout = self.loadout.copy()
                self.loadout = o_player.get_loadout()
                return True
            case 2: #Beer
                self.chamber[self.shotgun.n_bullet] = self.shotgun.current_bullet.copy()
                self.shotgun.shoot()
                self.n_bullet += 1
            case 3: #Burner Phone
                n_bullets = self.shotgun.get_n_bullets()
                if n_bullets < 2:
                    return False
                # ensure predicted position is within shotgun chamber bounds
                low = max(self.n_bullet + 1, 0)
                high = min(self.n_bullet + n_bullets, len(self.shotgun.chamber) - 1)
                if low > high:
                    return False
                predicted_bullet_pos = rnd.randint(low, high)
                predicted_bullet = self.shotgun.chamber[predicted_bullet_pos].copy()
                if predicted_bullet_pos < len(self.chamber):
                    self.chamber[predicted_bullet_pos] = predicted_bullet
            case 4: #Cigarette Pack
                if self.health == self.max_health:
                    return False
                self.health += 1
            case 5: #Expired Medicine
                if self.health == self.max_health:
                    return False
                if rnd.random() > 0.60:
                    if rnd.random() < 0.40:
                        self.health += 2
                    else:
                        self.health += 1
                else:
                    self.health -= 1
            case 6: #Hand saw
                self.shotgun.set_damage()
            case 7: #Handcuffs
                if self.can_use_handcuffs:
                    o_player.set_can_take_action(False)
                    self.set_can_use_handcuffs(False)
                else:
                    return False
            case 8: #Inverter
                self.shotgun.invert_bullet()
            case 9: #Magnifying Glass
                # guard index when copying from shotgun chamber
                self.chamber[self.n_bullet] = self.shotgun.get_current_bullet().copy()
        if not self.can_shoot:
            self.can_shoot = True
            self.loadout = self.copy_loadout.copy()
        return True