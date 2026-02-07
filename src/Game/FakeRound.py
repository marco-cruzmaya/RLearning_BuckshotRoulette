from Game.FakePlayer import FakePlayer
from Game.FakeShotgun import FakeShotgun
from Game.FakeLoadout import FakeLoadout
from Game.Round import Round
from Game.Bullet import Bullet
import random as rnd
class FakeRound(Round):

    def __init__(self,players = []):
        shotgun = self.set_shotgun()
        if players == []:
            players = self.set_players(shotgun)
        else:
            players = self.set_items(players,shotgun)
        super().__init__(players,shotgun)
        self.turns = [0]

    def set_shotgun(self):
        chamber = [Bullet(rnd.randint(2,3)) for i in range(rnd.randint(1,8))]
        shotgun = FakeShotgun(chamber)
        return shotgun
    
    def set_players(self,shotgun):
        players = []
        for i in range(2):
            items = [rnd.randint(1,6) for i in range(rnd.randint(1,9))]
            players.append(FakePlayer(rnd.randint(1,6),shotgun,FakeLoadout(items)))
        return players
    
    def set_items(self,players,shotgun):
        for i in range(2):
            items = [rnd.randint(1,6) for i in range(rnd.randint(1,9))]
            players[i].put_items(items)
            players[i].set_shotgun(shotgun)
        return players
    
    def set_turn_other(self):
        self.turns.append(0) if self.current_turn == 1 else self.turns.append(1)

    def set_turn(self):
        self.turns.append(self.current_turn)
    
    def get_turn(self):
        self.current_turn = self.turns.pop()
        return self.get_current(), self.get_other()
    
    def get_current(self):
        return self.players[self.current_turn]
    
    def get_other(self):
        return self.players[0] if self.current_turn == 1 else self.players[1]
    
    # action 0: shoot yourself
    # action 1: shoot other
    # action 2-9: use item in position i
    def play_turn(self,player,o_player,action):
        if not player.can_take_action:
            player.set_can_take_action(True)
            self.set_turn_other()
            return True
        if action >= 10:
            return False
        match action:
            case 0: #Shoot yourself
                if self.shotgun.current_bullet.get_type() == 3:
                    self.set_turn()
                else:
                    self.set_turn_other()
                action_status = player.shoot(player) #True if can_shoot and shooted else can_shoot was False
            case 1: #Shoot other
                action_status = player.shoot(o_player) #True if can_shoot and shooted else can_shoot was False
                self.set_turn_other()
            case _:
                action_status = player.use_item(pos = action-2, o_player = o_player)
                self.set_turn()

        if (not player.can_use_handcuffs) and o_player.can_take_action:
            player.set_can_use_handcuffs(True)
        return action_status