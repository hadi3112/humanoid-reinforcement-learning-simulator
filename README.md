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

Run headless (no GUI — much faster, recommended for actual training):

```bash
python train.py --no-render
```

Or with the GUI visible so you can watch training live (slow):

```bash
python train.py
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

---

## Full Workflow (Train → Evaluate)

```bash
# Step 1 — Train for 2 million steps (headless)
python train.py --no-render --timesteps 2000000

# Step 2 — Watch what it learned
python eval.py --slow
```

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

### viewer.py

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
| `--log-dir DIR` | TensorBoard log directory (default: ./logs/) |

### eval.py

| Flag | Description |
|---|---|
| `--model NAME` | Saved model to load (default: biped_ppo) |
| `--episodes N` | Number of episodes (0 = infinite) |
| `--urdf PATH` | URDF file to use |
| `--slow` | Slow-motion playback |
