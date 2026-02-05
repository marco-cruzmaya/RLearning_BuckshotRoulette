from Game.FakeRound import FakeRound


class Enviroment:
    def __init__(self):
        self.round = FakeRound()
    
    def step(self,action):
        if not self.round.status():
            if self.round.shotgun.is_empty():
                self.round = FakeRound(self.round.get_players())
            else:
                self.round = FakeRound()
        current_player, other_player = self.round.get_turn()
        obs = self.get_obs(current_player,other_player)
        actions_status = self.round.play_turn(current_player,other_player,action)
        n_obs = self.get_obs(current_player,other_player)
        return obs, action, n_obs, actions_status

    
    def get_obs(self,current_player,other_player):
        obs = [current_player.can_shoot,
               current_player.can_take_action,
               current_player.max_health,
               current_player.health,
               other_player.health,
               current_player.get_current_bullet()]
        obs += current_player.get_items()
        obs += list(current_player.get_bullets_count())
        obs += current_player.get_chamber()
        return obs