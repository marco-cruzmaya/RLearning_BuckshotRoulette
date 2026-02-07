from Game.FakeRound import FakeRound
import numpy as np
"""
Obs = [can_shoot,
       can_take_actions,
       can_use_handcuffs,
       max_health,
       player_health,
       enmy_health,
       current_type_bullet {0,1,2,3},
       item_pos_0,
       item_pos_1,
       item_pos_2,
       item_pos_3,
       item_pos_4,
       item_pos_5,
       item_pos_6,
       item_pos_7,
       n_life_rounds,
       n_blank_rounds,
       bullet_type_on_chamber_0,
       bullet_type_on_chamber_1,
       bullet_type_on_chamber_2,
       bullet_type_on_chamber_3,
       bullet_type_on_chamber_4,
       bullet_type_on_chamber_5,
       bullet_type_on_chamber_6,
       bullet_type_on_chamber_7]

Actions = { 0: Shoot yourself,
            1: Shoot other,
            2: Use item in pos 0,
            3: Use item in pos 1,
            4: Use item in pos 2,
            5: Use item in pos 3,
            6: Use item in pos 4,
            7: Use item in pos 4,
            8: Use item in pos 6,
            9: Use item in pos 7}
"""

class Enviroment:
    def __init__(self):
        self.round = FakeRound()
    
    def reset(self):
        if self.round.shotgun.is_empty():
            self.round = FakeRound(self.round.get_players())
        else:
            self.round = FakeRound()
        current_player, other_player = self.round.get_turn()
        return self.get_obs(current_player,other_player)

    
    def step(self,action):
        current_player = self.round.get_current()
        other_player = self.round.get_other()
        obs = self.get_obs(current_player,other_player)
        info = self.round.play_turn(current_player,other_player,action)
        next_obs = self.get_obs(current_player,other_player)
        done = not self.round.status()
        reward = self.set_reward(obs,action,next_obs,info)
        self.round.get_turn()
        return next_obs, reward, done, info
    
    def set_reward(self, obs, action, next_obs, info):
        # keep original reward logic; uses most-recent entries
        if not info:
            return -1.0
        match action:
            case 0:
                if obs[0] == 0:
                    return -1.0
                if obs[1] == 0:
                    return -1.0
                if obs[4] == next_obs[4]:
                    if obs[6] == 3:
                        return 0.75
                    else:
                        return 0.25
                if obs[4] > next_obs[4]:
                    if obs[6] == 2:
                        return -0.75
                    else:
                        return -0.5
            case 1:
                if obs[0] == 0:
                    return -1.0
                if obs[1] == 0:
                    return -1.0
                if obs[5] > next_obs[5]:
                    if obs[6] == 2:
                        return 0.75
                    else:
                        return 0.5
                if obs[5] == next_obs[5]:
                    if obs[6] == 1:
                        return -0.75
                    else:
                        return -0.25
        return 0.1

    
    def get_obs(self,current_player,other_player):
        obs = [current_player.can_shoot,
               current_player.can_take_action,
               current_player.can_use_handcuffs,
               current_player.max_health,
               current_player.health,
               other_player.health,
               current_player.get_current_bullet().get_type()]
        obs += current_player.get_items()
        obs += list(current_player.get_bullets_count())
        obs += current_player.get_chamber()
        return np.array(obs, dtype=np.float32)