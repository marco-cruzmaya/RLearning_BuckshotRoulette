import tensorflow as tf
import numpy as np
from typing import Sequence, Optional


class ActorCritic(tf.keras.Model):
    """Actor-Critic model with a shared MLP trunk, categorical policy head, and value head.

    Methods:
      - call(x): returns (logits, value)
      - action_value(obs): samples an action for a single observation and returns (action, logp, value)
      - get_logp_value(obs_batch, actions): returns (logp, value) for training
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (128, 128), activation: str = "tanh"):
        super().__init__()
        act = getattr(tf.nn, activation) if hasattr(tf.nn, activation) else tf.nn.tanh
        self.trunk = []
        for h in hidden_sizes:
            self.trunk.append(tf.keras.layers.Dense(h, activation=act))
        self.trunk = tf.keras.Sequential(self.trunk)

        self.logits_layer = tf.keras.layers.Dense(action_dim, name="policy_logits")
        self.value_layer = tf.keras.layers.Dense(1, name="value")

    @tf.function
    def call(self, x: tf.Tensor):
        """Forward pass. x shape: [B, obs_dim]"""
        h = self.trunk(x)
        logits = self.logits_layer(h)
        value = tf.squeeze(self.value_layer(h), axis=-1)
        return logits, value

    def action_value(self, obs: np.ndarray):
        """Sample an action for a single observation.

        Args:
            obs: 1D array of shape (obs_dim,) or a batch of shape (B, obs_dim).
        Returns:
            action (int), logp (float), value (float)
        """
        obs_np = np.asarray(obs, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np[None, :]
        logits, value = self.call(tf.convert_to_tensor(obs_np))
        logits = logits.numpy()
        value = value.numpy()
        # sample categorical action
        action = int(tf.random.categorical(tf.convert_to_tensor(logits), num_samples=1).numpy()[0, 0])
        logp_all = tf.nn.log_softmax(logits, axis=-1).numpy()
        logp = float(logp_all[0, action])
        return action, logp, float(value[0])

    def get_logp_value(self, obs_batch: np.ndarray, actions: np.ndarray):
        """Compute log-probs and values for a batch (for training).

        Returns:
            logp: shape (B,)
            value: shape (B,)
        """
        obs_b = tf.convert_to_tensor(np.asarray(obs_batch, dtype=np.float32))
        logits, values = self.call(obs_b)
        logp_all = tf.nn.log_softmax(logits, axis=-1)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        # gather log-prob of taken actions
        action_indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
        logp = tf.gather_nd(logp_all, action_indices)
        return logp, values


__all__ = ["ActorCritic"]
