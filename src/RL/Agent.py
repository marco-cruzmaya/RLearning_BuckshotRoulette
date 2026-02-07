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


from RL.RolloutBuffer import RolloutBuffer
from RL.Enviroment import Enviroment
import os
import time


class PPOAgent:
    def __init__(
        self,
        env: Enviroment,
        obs_dim: int,
        action_dim: int,
        hidden_sizes=(128, 128),
        lr: float = 1e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.25,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_envs: int = 1,
        logdir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        save_freq: int = 10000,
        use_gpu: bool = True,
        gpu_devices: Optional[Sequence[int]] = None,
    ):
        # configure GPU visibility before creating TF objects
        self.use_gpu = bool(use_gpu)
        try:
            if not self.use_gpu:
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    tf.config.set_visible_devices([], "GPU")
            else:
                # optionally restrict to a subset of GPU device indices
                if gpu_devices is not None:
                    gpus = tf.config.list_physical_devices("GPU")
                    chosen = [gpus[i] for i in gpu_devices if i < len(gpus)]
                    if chosen:
                        tf.config.set_visible_devices(chosen, "GPU")
                # enable memory growth for visible GPUs
                gpus = tf.config.list_physical_devices("GPU")
                for g in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(g, True)
                    except Exception:
                        pass
        except Exception:
            # if GPU configuration fails, continue with defaults
            pass

        self.env = env
        self.n_envs = int(n_envs)
        # create list of envs for vectorized collection
        self.envs = [env] + [Enviroment() for _ in range(self.n_envs - 1)]

        self.model = ActorCritic(obs_dim, action_dim, hidden_sizes)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-5)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # observation normalization
        self.obs_mean = np.zeros(obs_dim, dtype=np.float64)
        self.obs_var = np.ones(obs_dim, dtype=np.float64)
        self.obs_count = 1e-4

        # tensorboard and checkpoint
        self.logdir = logdir
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.writer = tf.summary.create_file_writer(logdir) if logdir else None
        if checkpoint_dir:
            ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            self.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
        else:
            self.ckpt_manager = None

            # metrics CSV path for training stats
            # default location (matches notebook lookup patterns)
            self.metrics_csv = os.path.join("training", "logs", "ppo_metrics.csv")
            # ensure parent dir exists
            try:
                os.makedirs(os.path.dirname(self.metrics_csv), exist_ok=True)
            except Exception:
                pass

    def _normalize_obs(self, obs: np.ndarray):
        """Normalize observations with running mean/var and clip."""
        obs = np.asarray(obs, dtype=np.float32)
        if self.obs_count <= 0:
            return obs
        mean = self.obs_mean.astype(np.float32)
        std = np.sqrt(self.obs_var / self.obs_count).astype(np.float32)
        return np.clip((obs - mean) / (std + 1e-8), -10.0, 10.0)

    def _update_obs_rms(self, obs_batch: np.ndarray):
        obs_batch = np.asarray(obs_batch, dtype=np.float64)
        batch_mean = np.mean(obs_batch, axis=0)
        batch_var = np.var(obs_batch, axis=0)
        batch_count = obs_batch.shape[0]

        # Welford-style update
        delta = batch_mean - self.obs_mean
        tot_count = self.obs_count + batch_count
        new_mean = self.obs_mean + delta * (batch_count / tot_count)
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.obs_count * batch_count / tot_count

        self.obs_mean = new_mean
        self.obs_var = M2 / tot_count
        self.obs_count = tot_count

    def collect_rollout(self, buf: RolloutBuffer, rollout_length: int):
        """Collect rollout_length steps from vectorized envs and store in buffer.

        Returns last observations array and last_dones array for each env.
        """
        # initialize obs for each env
        obs = [np.array(e.reset(), dtype=np.float32) for e in self.envs]
        ep_returns = [0.0 for _ in range(self.n_envs)]
        ep_lens = [0 for _ in range(self.n_envs)]
        last_dones = [False for _ in range(self.n_envs)]

        for _ in range(rollout_length):
            obs_batch = np.vstack(obs)
            # update normalization stats
            self._update_obs_rms(obs_batch)
            obs_norm = self._normalize_obs(obs_batch)

            # forward pass to get logits/values
            logits, values = self.model.call(tf.convert_to_tensor(obs_norm))
            logits_np = logits.numpy()
            values_np = values.numpy()

            # sample actions for each env
            acts = tf.random.categorical(tf.convert_to_tensor(logits_np), num_samples=1).numpy().flatten()
            logp_all = tf.nn.log_softmax(logits_np, axis=-1).numpy()
            logps = [float(logp_all[i, acts[i]]) for i in range(self.n_envs)]

            # step each env and append to buffer
            next_obs_list = []
            for i, e in enumerate(self.envs):
                next_obs, reward, done, info = e.step(int(acts[i]))
                buf.add(obs[i], int(acts[i]), float(reward), bool(done), float(values_np[i]), float(logps[i]))
                ep_returns[i] += float(reward)
                ep_lens[i] += 1
                next_obs_list.append(np.array(next_obs, dtype=np.float32))
                last_dones[i] = bool(done)
                if done:
                    # reset env immediately
                    next_obs_list[-1] = np.array(e.reset(), dtype=np.float32)

            obs = next_obs_list

        # compute last values for each env (bootstrap)
        obs_batch = np.vstack(obs)
        obs_norm = self._normalize_obs(obs_batch)
        _, last_values = self.model.call(tf.convert_to_tensor(obs_norm))
        last_values = last_values.numpy().astype(np.float32)

        # return per-env episode returns/lens observed during rollout so caller can log them
        return obs, last_values, last_dones, ep_returns, ep_lens

    @tf.function
    def _train_step(
        self,
        obs_b,
        act_b,
        old_logp_b,
        ret_b,
        adv_b,
        clip_eps: float,
        vf_coef: float,
        ent_coef: float,
        max_grad_norm: float,
    ):
        with tf.GradientTape() as tape:
            logits, values = self.model.call(obs_b)
            logp_all = tf.nn.log_softmax(logits, axis=-1)
            action_indices = tf.stack([tf.range(tf.shape(act_b)[0]), act_b], axis=1)
            new_logp = tf.gather_nd(logp_all, action_indices)

            ratio = tf.exp(new_logp - old_logp_b)
            surrogate1 = ratio * adv_b
            surrogate2 = tf.clip_by_value(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            value_loss = tf.reduce_mean(tf.square(ret_b - values)) * vf_coef

            probs = tf.nn.softmax(logits, axis=-1)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs * logp_all, axis=1))

            loss = policy_loss + value_loss - ent_coef * entropy

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        approx_kl = tf.reduce_mean(old_logp_b - new_logp)
        grad_norm = tf.linalg.global_norm(grads)
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "grad_norm": grad_norm,
        }

    def learn(
        self,
        total_timesteps: int,
        rollout_length: int = 2048,
        batch_size: int = 64,
        epochs: int = 10,
        verbose: bool = True,
        metrics_csv: Optional[str] = None,
        plot: bool = False,
    ):
        buf = RolloutBuffer(gamma=self.gamma, lam=self.lam)
        timesteps = 0
        episode_returns = []
        episode_lengths = []
        metrics = []

        if metrics_csv:
            self.metrics_csv = metrics_csv
            try:
                os.makedirs(os.path.dirname(self.metrics_csv), exist_ok=True)
            except Exception:
                pass

        while timesteps < total_timesteps:
            # collect rollout_length steps across vectorized envs
            obs_last, last_values, last_dones, ep_returns, ep_lens = self.collect_rollout(buf, rollout_length)
            timesteps += rollout_length * self.n_envs

            # compute GAE with last_values per env
            buf.compute_gae(last_values, n_envs=self.n_envs, last_dones=last_dones)

            # training loop over minibatches
            stats = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            n_updates = 0
            for obs_b, act_b, old_logp_b, ret_b, adv_b in buf.get_minibatches(batch_size, epochs):
                # normalize advantages per-mini-batch (stabilizes updates)
                try:
                    adv_b = (adv_b - np.mean(adv_b)) / (np.std(adv_b) + 1e-8)
                except Exception:
                    pass

                # normalize observations using running stats
                obs_b = self._normalize_obs(obs_b)

                obs_tf = tf.convert_to_tensor(obs_b, dtype=tf.float32)
                act_tf = tf.convert_to_tensor(act_b, dtype=tf.int32)
                old_logp_tf = tf.convert_to_tensor(old_logp_b, dtype=tf.float32)
                ret_tf = tf.convert_to_tensor(ret_b, dtype=tf.float32)
                adv_tf = tf.convert_to_tensor(adv_b, dtype=tf.float32)

                out = self._train_step(
                    obs_tf,
                    act_tf,
                    old_logp_tf,
                    ret_tf,
                    adv_tf,
                    self.clip_eps,
                    self.vf_coef,
                    self.ent_coef,
                    self.max_grad_norm,
                )
                for k in stats:
                    stats[k] += float(out[k])
                n_updates += 1

            # average stats over updates
            if n_updates > 0:
                for k in stats:
                    stats[k] /= n_updates

            # logging
            if verbose:
                print(
                    f"Timesteps={timesteps} updates={n_updates} avg_loss={stats['loss']:.4f} avg_policy={stats['policy_loss']:.4f} avg_value={stats['value_loss']:.4f} avg_entropy={stats['entropy']:.4f}"
                )

            # record metrics for CSV
            mean_return = float(np.mean(ep_returns)) if len(ep_returns) > 0 else float('nan')
            metrics.append(
                {
                    "timesteps": int(timesteps),
                    "updates": int(n_updates),
                    "loss": float(stats["loss"]),
                    "policy_loss": float(stats["policy_loss"]),
                    "value_loss": float(stats["value_loss"]),
                    "entropy": float(stats["entropy"]),
                    "mean_return": mean_return,
                }
            )

            # tensorboard
            if self.writer:
                with self.writer.as_default():
                    tf.summary.scalar("loss/total", stats["loss"], step=timesteps)
                    tf.summary.scalar("loss/policy", stats["policy_loss"], step=timesteps)
                    tf.summary.scalar("loss/value", stats["value_loss"], step=timesteps)
                    tf.summary.scalar("policy/entropy", stats["entropy"], step=timesteps)
                self.writer.flush()

            # checkpoint
            if self.ckpt_manager and timesteps % self.save_freq == 0:
                self.ckpt_manager.save()

            buf.clear()

        # save metrics to CSV after training
        try:
            self._save_metrics_csv(metrics, path=self.metrics_csv)
        except Exception as e:
            print('Failed to save metrics CSV:', e)

        # optionally plot
        if plot:
            try:
                self.plot_metrics(path=self.metrics_csv)
            except Exception as e:
                print('Failed to plot metrics:', e)

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """Restore model+optimizer from a checkpoint path or the latest in `checkpoint_dir`.

        Returns True if a checkpoint was restored.
        """
        cp = checkpoint_path
        if cp is None:
            if not self.checkpoint_dir:
                return False
            cp = tf.train.latest_checkpoint(self.checkpoint_dir)
        if not cp:
            return False
        try:
            ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            ckpt.restore(cp).expect_partial()
            print(f"Restored checkpoint: {cp}")
            return True
        except Exception as e:
            print(f"Failed to restore checkpoint {cp}: {e}")
            return False

    def _save_metrics_csv(self, metrics_list, path: Optional[str] = None):
        """Save list-of-dict metrics to CSV using pandas if available, else fallback to csv."""
        if path is None:
            path = self.metrics_csv
        if not path:
            return
        try:
            import pandas as pd

            df = pd.DataFrame(metrics_list)
            df.to_csv(path, index=False)
        except Exception:
            # fallback to csv
            import csv

            keys = set()
            for m in metrics_list:
                keys.update(m.keys())
            keys = sorted(keys)
            try:
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for m in metrics_list:
                        writer.writerow({k: m.get(k, "") for k in keys})
            except Exception as e:
                print('Could not write metrics CSV fallback:', e)

    def plot_metrics(self, path: Optional[str] = None):
        """Pretty plot training metrics from CSV using seaborn (or pandas plotting fallback)."""
        p = path or self.metrics_csv
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f"Metrics CSV not found: {p}")
        try:
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt

            df = pd.read_csv(p)
            sns.set(style="whitegrid")

            # Plot mean_return and losses on separate subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            if "timesteps" in df.columns:
                x = "timesteps"
            elif "updates" in df.columns:
                x = "updates"
            else:
                x = df.index

            if "mean_return" in df.columns:
                sns.lineplot(data=df, x=x, y="mean_return", ax=axes[0])
                axes[0].set_title("Mean Return over Training")

            loss_cols = [c for c in ("loss", "policy_loss", "value_loss") if c in df.columns]
            if loss_cols:
                df_loss = df.melt(id_vars=[x] if isinstance(x, str) else None, value_vars=loss_cols, var_name="loss_type", value_name="value")
                if isinstance(x, str):
                    sns.lineplot(data=df_loss, x=x, y="value", hue="loss_type", ax=axes[1])
                else:
                    sns.lineplot(data=df_loss, x=df_loss.index, y="value", hue="loss_type", ax=axes[1])
                axes[1].set_title("Losses over Training")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            # fallback: try simple pandas plotting
            try:
                import pandas as pd
                df = pd.read_csv(p)
                df.plot(subplots=True, figsize=(12, 8))
            except Exception as e2:
                print('Plotting failed:', e, e2)


__all__ = ["ActorCritic", "PPOAgent"]
