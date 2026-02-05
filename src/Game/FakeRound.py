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
        self.turns = [0,1]

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
    
    def set_turn(self):
        self.turn += [0,1]
    
    def get_turn(self):
        if self.turn == []:
            self.set_turn()
        self.current_turn = self.turn.pop()
        return self.get_current(), self.get_other()
    
    def get_current(self):
        return self.players[self.current_turn]
    
    def get_other(self):
        return self.players[0] if self.current_turn == 1 else self.players[1]
    
    # action 0: shoot yourself
    # action 1: shoot other
    # action 2-11: use item in position i
    def play_turn(self,player,o_player,action):
        if not player.can_take_action:
            player.set_can_take_action(True)
            return True
        if action >= 10:
            return False
        match action:
            case 0: #Shoot yourself
                return player.shoot(player) #True if can_shoot and shooted else can_shoot was False
            case 1: #Shoot other
                return player.shoot(o_player) #True if can_shoot and shooted else can_shoot was False
        return player.use_item(pos = action-2, o_player = o_player)