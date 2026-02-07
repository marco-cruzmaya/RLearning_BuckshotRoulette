from abc import ABC, abstractmethod

class Round(ABC):

    def __init__(self,players,shotgun):
        self.players = players
        self.shotgun = shotgun
        self.end_game = False
        self.turns = []
        self.current_turn = -1

    def status(self):
        flag = True
        for player in self.players:
            flag = flag and player.is_alive()
        flag = flag and (not self.shotgun.is_empty())
        return flag
    
    def get_players(self):
        return self.players
    
    @abstractmethod
    def play_turn(self,player,action):
        pass