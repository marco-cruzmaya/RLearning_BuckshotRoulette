import os
import time
import argparse
import numpy as np
import tensorflow as tf

from RL.Enviroment import Enviroment
from RL.Agent import ActorCritic

obs_names = ["can_shoot",
"can_take_actions",
"can_use_handcuffs",
"max_health",
"player_health",
"enmy_health",
"current_type_bullet",
"item_pos_0",
"item_pos_1",
"item_pos_2",
"item_pos_3",
"item_pos_4",
"item_pos_5",
"item_pos_6",
"item_pos_7",
"n_life_rounds",
"n_blank_rounds",
"bullet_type_on_chamber_0",
"bullet_type_on_chamber_1",
"bullet_type_on_chamber_2",
"bullet_type_on_chamber_3",
"bullet_type_on_chamber_4",
"bullet_type_on_chamber_5",
"bullet_type_on_chamber_6",
"bullet_type_on_chamber_7"]

items_name = {0:"None",
              1:"Adrenaline",
              2:"Beer",
              3:"Burner Phone",
              4:"Cigarrete Pack",
              5:"Expired Medicine",
              6:"Hand Saw",
              7:"Handcuffs",
              8:"Inverter",
              9:"Magnifying Glass"}

bullet_names = {0:"Uknown",
                1:"Used",
                2:"Life",
                3:"Blank"}

action_names = { 0: "Shoot yourself",
            1: "Shoot other",
            2: "Use item in pos 0",
            3: "Use item in pos 1",
            4: "Use item in pos 2",
            5: "Use item in pos 3",
            6: "Use item in pos 4",
            7: "Use item in pos 4",
            8: "Use item in pos 6",
            9: "Use item in pos 7"}


def load_model_if_available(obs_dim, action_dim, checkpoints_dir):
    latest = tf.train.latest_checkpoint(checkpoints_dir) if checkpoints_dir and os.path.isdir(checkpoints_dir) else None
    if not latest:
        print("No checkpoint found, will use random policy")
        return None

    # try several candidate hidden-size configurations to match the saved checkpoint
    candidates = [
        (128, 128),
        (obs_dim, obs_dim * 2, obs_dim * 4, obs_dim * 8, obs_dim * 4, obs_dim * 2, obs_dim, 128, 128),
    ]

    for hs in candidates:
        try:
            model = ActorCritic(obs_dim, action_dim, hidden_sizes=hs)
            # build model variables by running a dummy forward pass
            dummy = np.zeros((1, obs_dim), dtype=np.float32)
            model.call(tf.convert_to_tensor(dummy))

            ckpt = tf.train.Checkpoint(model=model)
            ckpt.restore(latest).expect_partial()
            print(f"Restored model from checkpoint: {latest} using hidden_sizes={hs}")
            return model
        except Exception as e:
            # try next candidate
            print(f"Failed to restore with hidden_sizes={hs}: {e}")

    print("Could not restore checkpoint into any candidate model; using random policy")
    return None


def pretty_obs(obs):
    # obs is a numpy array; show key fields (can_shoot, can_take_action, healths, bullet type)
    s = ""
    for i in range(len(obs_names)):
        if 15 > i >= 7:
            s += obs_names[i] + " : " + items_name[int(obs[i])] + "\n"
        elif i >= 17:
            s += obs_names[i] + " : " + bullet_names[int(obs[i])] + "\n"
        else:
            s += obs_names[i] + " : " + str(int(obs[i])) + "\n"
    return s


def run_episode(env, policy_model, action_dim, sleep=0.0, max_steps=500):
    obs = env.reset()
    done = False
    steps = 0
    total_reward = 0.0

    print("== New episode ==")
    print("Init:------------")

    while not done and steps < max_steps:
        print(env.round.shotgun)
        print(f"Player {env.round.current_turn}:\nstate=\n{pretty_obs(obs)}")
        if policy_model is not None:
            action, _, _ = policy_model.action_value(obs)
            # ensure action is a Python int
            if hasattr(action, "numpy"):
                action = int(action.numpy()) 
            else:
                action = int(action)
            print(f"Chosen action: {action} -> {action_names.get(action, 'Unknown')}")
        else:
            action = int(np.random.randint(0, action_dim))
            print(f"Random action: {action} -> {action_names.get(action, 'Unknown')}")

        next_obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        print(f"Step {steps:03d}:\n reward={reward:.2f} done={done}\nnext_state=\n{pretty_obs(next_obs)}")
        obs = next_obs
        steps += 1
        if sleep > 0.0 and reward != -1.0:
            time.sleep(sleep)

    print(f"Episode finished steps={steps} total_reward={total_reward:.2f}\n")
    return total_reward, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between steps for readability")
    parser.add_argument("--checkpoints", type=str, default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    parser.add_argument("--action-dim", type=int, default=10)
    args = parser.parse_args()

    env = Enviroment()
    obs = env.reset()
    obs_dim = len(obs)
    action_dim = args.action_dim

    model = load_model_if_available(obs_dim, action_dim, args.checkpoints)

    ep_rewards = []
    for e in range(args.episodes):
        r, steps = run_episode(env, model, action_dim, sleep=args.sleep)
        ep_rewards.append(r)

    print(f"Summary: episodes={len(ep_rewards)} avg_reward={np.mean(ep_rewards):.3f} rewards={ep_rewards}")


if __name__ == "__main__":
    main()
