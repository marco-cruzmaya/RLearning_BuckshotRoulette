import numpy as np

"""
Obs = [can_shoot,
       can_take_actions,
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

class State:
    """Lightweight trajectory container (kept for compatibility).

    Note: avoid mutable defaults; use lists created per-instance.
    """
    def __init__(self, obs=None, actions=None, n_obs=None, rewards=None):
        self.obs = [] if obs is None else list(obs)
        self.actions = [] if actions is None else list(actions)
        self.n_obs = [] if n_obs is None else list(n_obs)
        self.rewards = [] if rewards is None else list(rewards)

    def append(self, obs, action, n_obs, action_status):
        self.obs.append(obs)
        self.actions.append(action)
        self.n_obs.append(n_obs)
        self.rewards.append(self.set_reward(action_status))

    def get_batch(self, gamma=0.99):
        returns = self.discount_rewards(gamma)
        return np.array(self.obs, dtype=np.float32), np.array(self.actions, dtype=np.int32), np.array(returns, dtype=np.float32)

    def discount_rewards(self, gamma=0.99):
        returns = []
        r = 0.0
        for reward in reversed(self.rewards):
            r = reward + gamma * r
            returns.append(r)
        return list(reversed(returns))

    def set_reward(self, action_status):
        # keep original reward logic; uses most-recent entries
        last_obs = self.obs[-1]
        last_action = self.actions[-1]
        last_n_obs = self.n_obs[-1]
        if not action_status:
            return -1.0
        match last_action:
            case 0:
                if last_obs[0] == 0:
                    return -1.0
                if last_obs[1] == 0:
                    return -1.0
                if last_obs[3] == last_n_obs[3]:
                    if last_obs[5] == 3:
                        return 0.75
                    else:
                        return 0.25
                if last_obs[3] > last_n_obs[3]:
                    if last_obs[5] == 2:
                        return -0.75
                    else:
                        return -0.5
            case 1:
                if last_obs[0] == 0:
                    return -1.0
                if last_obs[1] == 0:
                    return -1.0
                if last_obs[4] > last_n_obs[4]:
                    if last_obs[5] == 2:
                        return 0.75
                    else:
                        return 0.5
                if last_obs[4] == last_n_obs[4]:
                    if last_obs[5] == 1:
                        return -0.75
                    else:
                        return -0.25
        return 0.1


class RolloutBuffer:
    """Simple rollout buffer for on-policy algorithms (PPO).

    Usage:
      buf = RolloutBuffer()
      buf.add(obs, action, reward, done, value, logp)
      ... after rollout ... buf.compute_gae(last_value)
      for mb in buf.get_minibatches(batch_size, epochs): use minibatch
    """
    def __init__(self, gamma=0.99, lam=0.95):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = None
        self.returns = None
        self.gamma = gamma
        self.lam = lam

    def add(self, obs, action, reward, done, value, logp):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.log_probs.append(float(logp))

    def compute_gae(self, last_value, gamma=None, lam=None):
        """Compute GAE advantages and returns for the collected rollout.

        last_value: bootstrap value for the timestep after the final step
        """
        if gamma is None:
            gamma = self.gamma
        if lam is None:
            lam = self.lam

        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                nextnonterminal = 1.0 - float(self.dones[t])
                nextvalues = float(last_value)
            else:
                nextnonterminal = 1.0 - float(self.dones[t + 1])
                nextvalues = float(self.values[t + 1])
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - float(self.values[t])
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        self.advantages = advantages
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)

        # normalize advantages
        adv_mean = np.mean(self.advantages) if self.advantages.size > 0 else 0.0
        adv_std = np.std(self.advantages) if self.advantages.size > 0 else 1.0
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

    def get(self):
        return (
            np.array(self.obs, dtype=np.float32),
            np.array(self.actions, dtype=np.int32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.returns, dtype=np.float32),
            np.array(self.advantages, dtype=np.float32),
        )

    def get_minibatches(self, batch_size, epochs=4):
        data_size = len(self.rewards)
        if data_size == 0:
            return
        inds = np.arange(data_size)
        obs_all, actions_all, logp_all, ret_all, adv_all = self.get()
        for _ in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, data_size, batch_size):
                mb_inds = inds[start : start + batch_size]
                yield (
                    obs_all[mb_inds],
                    actions_all[mb_inds],
                    logp_all[mb_inds],
                    ret_all[mb_inds],
                    adv_all[mb_inds],
                )

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = None
        self.returns = None