# Bipedal Robot Walking — Reinforcement Learning

Train a bipedal robot to walk using **PPO** (Proximal Policy Optimization) in a **PyBullet** physics simulation, or just open the viewer and play with the joints yourself.

Built with [PyBullet](https://pybullet.org/), [Gymnasium](https://gymnasium.farama.org/), and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

---

## Setup

```bash
git clone https://github.com/Einsbon/bipedal-robot-walking-simulation.git
cd bipedal-robot-walking-simulation
pip install -r requirements.txt
```

> **Windows users:** PyBullet compiles from source, so you need
> [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> installed first (select "Desktop development with C++").

---

## Quick Start

### 1. Just view the robot (no training needed)

Open the PyBullet GUI with a slider for every joint — drag them to move the robot in real time.

```bash
python viewer.py
```

Use your own URDF:

```bash
python viewer.py --urdf humanoid_leg_12dof.8.urdf
```

Pin the robot in the air so it doesn't fall while you inspect joints:

```bash
python viewer.py --fixed
```

Zero gravity sandbox:

```bash
python viewer.py --no-gravity
```

### 2. Train a walking policy

#### Headless Parallel Training (Fastest, recommended)
Run multiple environments in parallel across your CPU cores without any GUI:

```bash
python train.py --no-render --num-envs 8
```

#### Parallel Training with Single-GUI Monitoring
Run in parallel but open exactly **one** PyBullet GUI window (for the first environment) so you can watch progress live:

```bash
python train.py --num-envs 4
```

This saves the trained model to `biped_ppo.zip` when finished.

### 3. Watch the trained robot walk

After training is done, run inference to see the result:

```bash
python eval.py
```

Slow-motion playback:

```bash
python eval.py --slow
```

Run a specific number of episodes:

```bash
python eval.py --episodes 10
```

### 4. Monitor training with TensorBoard

```bash
tensorboard --logdir ./logs/
```

#### Deciphering TensorBoard Metrics

When monitoring training in TensorBoard, focus on these key graphs under the `rollout/` and `train/` sections:

* **`rollout/ep_rew_mean` (Mean Episode Reward):** 
  * *What to look for:* A steady, upward logarithmic curve. 
  * *Interpretation:* Represents overall walking performance. Higher is better. A flatline means the robot has collapsed or is stuck in a local minimum (e.g., crouching statically).
* **`rollout/ep_len_mean` (Mean Episode Length):** 
  * *What to look for:* Climbing from very low numbers (20-50 steps) towards the maximum limit (`1000`).
  * *Interpretation:* Represents survival time. When it reaches 1000, the robot successfully stands and walks for the entire duration of the episode without falling.
* **`train/entropy_loss` (Policy Entropy):** 
  * *What to look for:* A gradual downward slope (becoming less negative).
  * *Interpretation:* Measures randomness/exploration. Starts high (random flailing) and should decline as the policy becomes confident in its actions.
  * *Warning:* If it drops to zero too fast (e.g. within 50k steps), the robot has prematurely converged (e.g., locking its joints).
* **`train/value_loss` (Value Loss):** 
  * *What to look for:* Spikes early on, but stabilizes and trends downwards.
  * *Interpretation:* Shows how well the Critic predicts rewards. Lower means the Critic has a highly accurate model of physical dynamics.

---

## Full Workflow (Train → Evaluate)

```bash
# Step 1 — Train for 2 million steps (headless, 4 parallel environments)
python train.py --no-render --num-envs 4 --timesteps 2000000

# Step 2 — Watch what it learned
python eval.py --slow
```
*Note: on an 8GB RAM Intel Core i5 8th Gen HP Probook 450 G6, the 2 million timesteps were computed within 2 hrs, going beyond 1,000,000 time steps during Training would be Overkill for this Simulation, unless you have better CPU Cores, or an RTX 4060 GPU for running multiple environments for training in Paralle. The commands for that are give below in CLI Options

---

## Project Structure

| File | What it does |
|---|---|
| `env.py` | Gymnasium environment wrapping PyBullet (observation, reward, physics) |
| `train.py` | PPO training script |
| `eval.py` | Load a trained model and watch it in the GUI |
| `viewer.py` | Standalone URDF viewer with joint sliders — no RL needed |
| `requirements.txt` | Python dependencies |

---

## CLI Options

| Flag | Description |
|---|---|
| `--urdf PATH` | URDF file to load |
| `--fixed` | Pin the robot base in the air |
| `--no-gravity` | Disable gravity |
| `--height N` | Spawn height in metres (default: 1.0) |
| `--slow` | Real-time speed instead of max speed |

### train.py

| Flag | Description |
|---|---|
| `--no-render` | Headless training (no GUI — fast) |
| `--timesteps N` | Total training steps (default: 2,000,000) |
| `--model-name NAME` | Save filename (default: biped_ppo) |
| `--urdf PATH` | URDF file to use |
| `--learning-rate LR` | PPO learning rate (default: 3e-4) |
| `--num-envs N` | Number of parallel environments (default: 4) |
| `--log-dir DIR` | TensorBoard log directory (default: ./logs/) |

### eval.py

| Flag | Description |
|---|---|
| `--model NAME` | Saved model to load (default: biped_ppo) |
| `--episodes N` | Number of episodes (0 = infinite) |
| `--urdf PATH` | URDF file to use |
| `--slow` | Slow-motion playback |
