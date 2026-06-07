"""
Train a PPO agent on the Bipedal Walking Environment.

Usage
-----
    python train.py                          # Train with GUI visible (slow, for debugging)
    python train.py --no-render              # Train headless (fast, recommended for real training)
    python train.py --timesteps 5000000      # Custom timestep count
    python train.py --model-name my_biped    # Custom save name

TensorBoard
-----------
    tensorboard --logdir ./logs/
"""

import argparse
from stable_baselines3 import PPO
from env import BipedEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO on PyBullet Biped Walking Environment"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total training timesteps (default: 2,000,000)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable GUI — headless training (much faster)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="biped_ppo",
        help="Filename for saved model (default: biped_ppo)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/",
        help="TensorBoard log directory (default: ./logs/)",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="biped/biped2d_pybullet.urdf",
        help="URDF file path (searched in pybullet_data and working directory)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate (default: 3e-4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    render_mode = "direct" if args.no_render else "human"

    # ---- Banner ---- #
    print("=" * 60)
    print("  Biped PPO Training")
    print("=" * 60)
    print(f"  Render mode   : {render_mode}")
    print(f"  URDF          : {args.urdf}")
    print(f"  Timesteps     : {args.timesteps:,}")
    print(f"  Learning rate : {args.learning_rate}")
    print(f"  Model name    : {args.model_name}")
    print(f"  Log directory : {args.log_dir}")
    print("=" * 60)

    # ---- Create environment ---- #
    env = BipedEnv(render_mode=render_mode, urdf_path=args.urdf)

    print(f"\n  Joints found  : {env.num_joints}")
    print(f"  Joint names   : {list(env.joint_names.values())}")
    print(f"  Obs space     : {env.observation_space.shape}")
    print(f"  Action space  : {env.action_space.shape}")
    print("=" * 60)

    # ---- Create PPO agent ---- #
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.log_dir,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="auto",
    )

    # ---- Train ---- #
    print("\nStarting training...\n")
    model.learn(total_timesteps=args.timesteps)

    # ---- Save ---- #
    model.save(args.model_name)
    print(f"\nModel saved to: {args.model_name}.zip")

    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
