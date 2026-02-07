import sys
import os
import argparse

# ensure 'src' is on sys.path so we can import the package modules
from RL.Enviroment import Enviroment
from RL.Agent import PPOAgent
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true", help="If set, restore latest checkpoint and continue training")
    parser.add_argument("--no-gpu", action="store_true", help="If set, force CPU-only training")
    parser.add_argument("--total-timesteps", type=int, default=256 * 100)
    args = parser.parse_args()

    env = Enviroment()
    obs = env.reset()
    obs_dim = len(obs)
    action_dim = 10

    logdir = os.path.join(os.path.dirname(__file__), "logs")
    ckdir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckdir, exist_ok=True)

    # create agent, allow disabling GPU
    agent = PPOAgent(
        env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_envs=4,
        logdir=logdir,
        checkpoint_dir=ckdir,
        save_freq=512,
        use_gpu=not args.no_gpu,
        hidden_sizes= [obs_dim,obs_dim*2,obs_dim*4,obs_dim*8,obs_dim*4,obs_dim*2,obs_dim,128,128]
    )

    if args.finetune:
        restored = agent.load_checkpoint()
        if not restored:
            print("No checkpoint restored; starting training from scratch")

    print("Starting longer smoke training with tensorboard and checkpoints")
    try:
        agent.learn(total_timesteps=args.total_timesteps, rollout_length=64, batch_size=64, epochs=10, verbose=1)
    except Exception as e:
        print("Training run failed:", e)
    else:
        print("Training run completed")


if __name__ == "__main__":
    main()
