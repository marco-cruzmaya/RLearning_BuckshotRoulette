import numpy as np

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

    def compute_gae(self, last_values, n_envs: int = 1, last_dones=None, gamma=None, lam=None):
        """Compute GAE advantages and returns for the collected rollout.

        Supports interleaved buffers collected from `n_envs` parallel environments.

        Args:
            last_values: scalar or array-like of length `n_envs` with bootstrap values
            n_envs: number of parallel environments used when collecting the buffer
            last_dones: optional list/array of booleans indicating whether each env was done
        """
        if gamma is None:
            gamma = self.gamma
        if lam is None:
            lam = self.lam

        if last_dones is None:
            last_dones = [False] * n_envs

        # ensure numpy arrays
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.bool_)
        rewards = np.array(self.rewards, dtype=np.float32)

        # allow last_values to be scalar or array
        last_values_arr = np.repeat(float(last_values), n_envs) if np.isscalar(last_values) else np.asarray(last_values, dtype=np.float32)
        last_dones_arr = np.asarray(last_dones, dtype=np.bool_)

        lastgaelam = 0.0
        for t in reversed(range(n)):
            env_idx = t % n_envs
            next_idx = t + n_envs
            if next_idx < n:
                nextnonterminal = 1.0 - float(dones[next_idx])
                nextvalues = float(values[next_idx])
            else:
                nextnonterminal = 1.0 - float(last_dones_arr[env_idx])
                nextvalues = float(last_values_arr[env_idx])

            delta = rewards[t] + gamma * nextvalues * nextnonterminal - float(values[t])
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        self.advantages = advantages
        self.returns = self.advantages + values

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