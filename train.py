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
from stable_baselines3.common.vec_env import SubprocVecEnv


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
        "--load-model",
        type=str,
        default="",
        help="Path to an existing model zip file to load and resume training (default: empty, train from scratch)",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="biped2d_pybullet.urdf",
        help="URDF file path (searched in pybullet_data and working directory)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments to run (default: 4)",
    )
    return parser.parse_args()


def make_env(rank, urdf_path, no_render):
    """Helper to instantiate a BipedEnv in a separate subprocess."""
    def _init():
        # All environments use render_mode="direct" to satisfy Stable-Baselines3,
        # but rank 0 overrides PyBullet to open a GUI window if no_render is False
        force_gui = (not no_render) and (rank == 0)
        return BipedEnv(render_mode="direct", urdf_path=urdf_path, force_gui=force_gui)
    return _init


def main():
    args = parse_args()

    # ---- Banner ---- #
    print("=" * 60)
    print("  Biped PPO Training (Parallelized)")
    print("=" * 60)
    print(f"  Parallel Envs : {args.num_envs}")
    print(f"  GUI Visible   : {not args.no_render} (Rank 0 only)")
    print(f"  URDF          : {args.urdf}")
    print(f"  Timesteps     : {args.timesteps:,}")
    print(f"  Learning rate : {args.learning_rate}")
    if args.load_model:
        print(f"  Load Model    : {args.load_model}")
    else:
        print("  Load Model    : None (training from scratch)")
    print(f"  Model name    : {args.model_name}")
    print(f"  Log directory : {args.log_dir}")
    print("=" * 60)

    # ---- Create environment ---- #
    if args.num_envs > 1:
        env_fns = [make_env(i, args.urdf, args.no_render) for i in range(args.num_envs)]
        env = SubprocVecEnv(env_fns)
        num_joints = env.get_attr("num_joints")[0]
        joint_names = env.get_attr("joint_names")[0]
    else:
        render_mode = "direct" if args.no_render else "human"
        env = BipedEnv(render_mode=render_mode, urdf_path=args.urdf)
        num_joints = env.num_joints
        joint_names = env.joint_names

    print(f"\n  Joints found  : {num_joints}")
    print(f"  Joint names   : {list(joint_names.values())}")
    print(f"  Obs space     : {env.observation_space.shape}")
    print(f"  Action space  : {env.action_space.shape}")
    print("=" * 60)

    # ---- Create or load PPO agent ---- #
    if args.load_model:
        load_path = args.load_model
        if load_path.endswith(".zip"):
            load_path = load_path[:-4]
        print(f"\nLoading existing model checkpoint from: {load_path}.zip")
        
        from stable_baselines3.common.utils import get_schedule_fn
        lr_schedule = get_schedule_fn(args.learning_rate)
        custom_objects = {
            "learning_rate": args.learning_rate,
            "lr_schedule": lr_schedule,
        }
        model = PPO.load(load_path, env=env, tensorboard_log=args.log_dir, custom_objects=custom_objects)
        
        # Explicitly apply to model properties and optimizer param groups
        model.learning_rate = args.learning_rate
        model.lr_schedule = lr_schedule
        for param_group in model.policy.optimizer.param_groups:
            param_group["lr"] = args.learning_rate
    else:
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
    try:
        model.learn(total_timesteps=args.timesteps)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model state...")
    finally:
        # ---- Save ---- #
        model.save(args.model_name)
        print(f"\nModel saved to: {args.model_name}.zip")
        env.close()
        print("Training terminated/completed cleanly.")


if __name__ == "__main__":
    main()
