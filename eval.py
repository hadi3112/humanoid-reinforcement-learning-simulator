"""
Evaluate / visualise a trained PPO policy on the Biped Walking Environment.

Usage
-----
    python eval.py                           # Load default model and watch
    python eval.py --model biped_ppo         # Load a specific model
    python eval.py --episodes 10             # Run exactly 10 episodes
    python eval.py --slow                    # Slow-motion playback (60 FPS cap)

PyBullet GUI Controls
---------------------
    Mouse drag      : rotate camera
    Scroll wheel    : zoom in / out
    Ctrl + drag     : pan camera
"""

import argparse
import time

from stable_baselines3 import PPO
from env import BipedEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained PPO on Biped Walking Environment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="biped_ppo",
        help="Path to saved model (without .zip extension)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run (0 = infinite loop)",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="biped/biped2d_pybullet.urdf",
        help="URDF file path (searched in pybullet_data and working directory)",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Slow-motion playback — caps at ~60 FPS for easier viewing",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Banner ---- #
    print("=" * 60)
    print("  Biped PPO Evaluation")
    print("=" * 60)
    print(f"  Model         : {args.model}")
    print(f"  URDF          : {args.urdf}")
    print(f"  Episodes      : {'infinite' if args.episodes == 0 else args.episodes}")
    print(f"  Slow motion   : {args.slow}")
    print(f"  Deterministic : {args.deterministic}")
    print("=" * 60)

    # ---- Create environment (always GUI for evaluation) ---- #
    env = BipedEnv(render_mode="human", urdf_path=args.urdf)

    # ---- Load trained model ---- #
    model = PPO.load(args.model)
    print(f"\n  Model loaded from: {args.model}.zip")
    print("  Watching trained policy...  (Ctrl+C to quit)\n")

    episode = 0
    try:
        while True:
            obs, _ = env.reset()
            episode += 1
            episode_reward = 0.0
            step = 0

            while True:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step += 1

                if args.slow:
                    time.sleep(1.0 / 60.0)

                if terminated or truncated:
                    height = info.get("torso_height", 0.0)
                    print(
                        f"  Episode {episode:4d}  |  "
                        f"Steps: {step:5d}  |  "
                        f"Reward: {episode_reward:9.2f}  |  "
                        f"Torso Height: {height:.3f}"
                    )
                    break

            # Stop after N episodes if requested
            if 0 < args.episodes <= episode:
                break

    except KeyboardInterrupt:
        print("\n\n  Stopped by user.")

    finally:
        env.close()
        print(f"\n  Total episodes evaluated: {episode}")


if __name__ == "__main__":
    main()
