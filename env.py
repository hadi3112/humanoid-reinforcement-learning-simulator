"""
Bipedal Robot Walking Environment for Reinforcement Learning.

Uses PyBullet physics engine with a Gymnasium-compatible interface.
Designed for training with Stable-Baselines3 PPO.

Author: Built on top of Einsbon's bipedal-robot-walking-simulation
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data


class BipedEnv(gym.Env):
    """
    PyBullet Bipedal Robot Walking Environment.

    Observation space (all normalized to [-1, 1]):
        - Joint positions              (num_joints)
        - Joint velocities             (num_joints)
        - Base position                (3: x, y, z)
        - Base orientation, Euler      (3: roll, pitch, yaw)
        - Base linear velocity         (3: vx, vy, vz)
        - Base angular velocity        (3: wx, wy, wz)
        Total: num_joints * 2 + 12

    Action space:
        - Normalized torques [-1, 1] for each actuated joint,
          scaled internally to [-max_torque, max_torque].

    Reward:
        forward_velocity - energy_penalty - fall_penalty
    """

    metadata = {"render_modes": ["human", "direct"], "render_fps": 60}

    def __init__(
        self,
        render_mode="human",
        urdf_path="biped2d_pybullet.urdf",
        max_episode_steps=1000,
        max_torque=20.0,
        fall_threshold=None,
        initial_height=None,
        force_gui=False,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.urdf_path = urdf_path
        self.max_episode_steps = max_episode_steps
        self.max_torque = max_torque
        
        # Determine appropriate initial spawn height and fall threshold based on URDF
        if initial_height is None:
            if "biped2d" in self.urdf_path:
                self.initial_height = -0.388
            elif "12dof" in self.urdf_path:
                self.initial_height = 0.31
            else:
                self.initial_height = 1.0
        else:
            self.initial_height = initial_height

        if fall_threshold is None:
            if "biped2d" in self.urdf_path:
                self.fall_threshold = 0.55
            elif "12dof" in self.urdf_path:
                self.fall_threshold = 0.15
            else:
                self.fall_threshold = 0.3
        else:
            self.fall_threshold = fall_threshold

        self.step_count = 0
        self.prev_action = None

        # ---- Connect to PyBullet ---- #
        if self.render_mode == "human" or force_gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 1, physicsClientId=self.physics_client
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        # ---- Physics parameters ---- #
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # ---- Load ground plane (flat, no bumps) ---- #
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        # ---- Load robot ---- #
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, self.initial_height],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            physicsClientId=self.physics_client,
        )

        # ---- Discover actuated joints ---- #
        self.joint_ids = []
        self.joint_names = {}
        self._discover_joints()
        self.num_joints = len(self.joint_ids)

        # ---- Discover torso link ---- #
        self._discover_torso_link()

        if self.num_joints == 0:
            raise ValueError(
                f"No actuated joints found in URDF: {self.urdf_path}. "
                "Ensure the URDF contains revolute or prismatic joints."
            )

        # Disable default velocity motors — required for torque control
        self._disable_default_motors()

        # ---- Action space ---- #
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32,
        )

        # ---- Observation space ---- #
        # joint_pos + joint_vel + base_pos + base_euler + base_lin_vel + base_ang_vel
        obs_dim = self.num_joints * 2 + 12
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # ---- Normalization scales ---- #
        self._joint_pos_scale = np.pi        # radians
        self._joint_vel_scale = 10.0         # rad/s
        self._base_pos_scale = np.array([10.0, 10.0, 2.0])  # metres
        self._base_euler_scale = np.pi       # radians
        self._base_lin_vel_scale = 5.0       # m/s
        self._base_ang_vel_scale = 10.0      # rad/s

        self.prev_action = np.zeros(self.num_joints, dtype=np.float32)
        self.goal_x = 20.0
        self.prev_dist_to_goal = self.goal_x

    # ================================================================== #
    #  Internal helpers
    # ================================================================== #

    def _discover_joints(self):
        """Find all actuated revolute and prismatic joints in the loaded URDF, excluding virtual constraint joints."""
        self.joint_ids = []
        self.joint_names = {}
        virtual_joints = ["y_to_world", "z_to_y", "torso_to_z"]
        total = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        for i in range(total):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            jtype = info[2]
            jname = info[1].decode("utf-8")
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                if not any(v in jname for v in virtual_joints):
                    self.joint_ids.append(i)
                    self.joint_names[i] = jname

    def _discover_torso_link(self):
        """Find the link ID representing the main torso/body of the robot."""
        self.torso_link_id = -1
        total = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        for i in range(total):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            child_name = info[12].decode("utf-8").lower()
            if child_name in ["body", "torso", "pelvis", "waist"]:
                self.torso_link_id = i
                break

    def _get_torso_height(self):
        """Get the height (Z-coordinate) of the torso/body link, falling back to base height if not found."""
        if self.torso_link_id == -1:
            pos, _ = p.getBasePositionAndOrientation(
                self.robot_id, physicsClientId=self.physics_client
            )
            return pos[2]
        else:
            state = p.getLinkState(
                self.robot_id, self.torso_link_id, physicsClientId=self.physics_client
            )
            return state[0][2]

    def _disable_default_motors(self):
        """Set default motor forces to zero for all joints in the robot so passive/virtual joints can move freely and active joints are not resisted."""
        total = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        for i in range(total):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self.physics_client,
            )

    @staticmethod
    def _normalise(value, scale):
        """Element-wise clip(value / scale, -1, 1)."""
        return np.clip(np.asarray(value, dtype=np.float32) / scale, -1.0, 1.0)

    # ================================================================== #
    #  Observation
    # ================================================================== #

    def _get_obs(self):
        """Build the fully-normalised observation vector."""
        joint_pos = []
        joint_vel = []
        for j in self.joint_ids:
            state = p.getJointState(
                self.robot_id, j, physicsClientId=self.physics_client
            )
            joint_pos.append(state[0])
            joint_vel.append(state[1])

        base_pos, base_quat = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        base_euler = p.getEulerFromQuaternion(base_quat)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )

        obs = np.concatenate(
            [
                self._normalise(joint_pos, self._joint_pos_scale),
                self._normalise(joint_vel, self._joint_vel_scale),
                self._normalise(base_pos, self._base_pos_scale),
                self._normalise(base_euler, self._base_euler_scale),
                self._normalise(base_lin_vel, self._base_lin_vel_scale),
                self._normalise(base_ang_vel, self._base_ang_vel_scale),
            ]
        ).astype(np.float32)

        return obs

    # ================================================================== #
    #  Reward
    # ================================================================== #

    def _compute_reward(self, action):
        """
        reward = progress_reward + goal_reward + alive_bonus - energy_penalty - fall_penalty - crouch_penalty - symmetry_penalty - ankle_penalty - knee_penalty - torso_tilt_penalty

        Components
        ----------
        progress_reward : float
            Reward for making forward progress towards goal_x. Weighted by 3.0 to prioritize forward walking.
        goal_reward : float
            Big bonus (+100.0) awarded once upon reaching goal_x.
        alive_bonus : float
            Encourages the robot to stay standing (+0.5 per step).
        energy_penalty : float
            0.001 * sum(action^2)  — discourages wasteful torques.
        fall_penalty : float
            -100 applied once when the robot's torso drops below the fall_threshold.
        crouch_penalty : float
            Penalises both knees bending simultaneously to prevent crouching static states.
        symmetry_penalty : float
            Penalises in-phase hip movement to encourage out-of-phase leg swing.
        ankle_penalty : float
            Stiffens the ankles, penalizing flexion beyond minimal angles.
        knee_penalty : float
            Penalizes forward knee bending/extension past 0.
        torso_tilt_penalty : float
            Penalizes torso tilt away from upright.
        """
        # Get torso and base info
        torso_height = self._get_torso_height()
        base_pos, base_quat = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )

        # Resolve 2D planar robot progress tracking: Y for biped2d, X otherwise.
        if "biped2d" in self.urdf_path:
            torso_state = p.getLinkState(self.robot_id, self.torso_link_id, physicsClientId=self.physics_client)
            base_x = torso_state[0][1]
            torso_quat = torso_state[1]
            torso_euler = p.getEulerFromQuaternion(torso_quat)
            pitch = torso_euler[0]
        else:
            base_x = base_pos[0]
            base_euler = p.getEulerFromQuaternion(base_quat)
            pitch = base_euler[1]

        # 1. Goal progress reward (prioritized and scaled)
        dist_to_goal = abs(self.goal_x - base_x)
        progress = self.prev_dist_to_goal - dist_to_goal
        # Scale progress to match velocity units (m/s) and multiply by 3.0 to prioritize forward movement
        progress_reward = 3.0 * float(progress / self.time_step)
        self.prev_dist_to_goal = dist_to_goal

        # Goal reached bonus
        goal_reward = 0.0
        if base_x >= self.goal_x:
            goal_reward = 100.0

        # Alive bonus
        alive_bonus = 0.5

        # 2. Crouching & Symmetry Penalties
        # Read joint states
        joint_states = {}
        for jid, name in self.joint_names.items():
            state = p.getJointState(self.robot_id, jid, physicsClientId=self.physics_client)
            joint_states[name] = state[0]

        r_knee = joint_states.get("r_knee", 0.0)
        l_knee = joint_states.get("l_knee", 0.0)
        r_hip = joint_states.get("torso_to_rightleg", 0.0)
        l_hip = joint_states.get("torso_to_leftleg", 0.0)
        r_ankle = joint_states.get("r_ankle", 0.0)
        l_ankle = joint_states.get("l_ankle", 0.0)

        # Penalise both knees bending at the same time (double-knee buckling / crouching)
        crouch_penalty = 2.0 * float(abs(r_knee * l_knee))
        
        # Penalise in-phase hip movement (encourage opposite swing of hips)
        symmetry_penalty = 1.0 * float((r_hip + l_hip) ** 2)

        # Restrict ankle joint rotation to minimal flexion (stiffen ankles)
        ankle_penalty = 15.0 * (r_ankle**2 + l_ankle**2)

        # Prevent forward knee bending (extension past 0 degrees)
        knee_penalty = 10.0 * (max(0.0, -r_knee)**2 + max(0.0, -l_knee)**2)

        # Penalize torso tilt
        torso_tilt_penalty = 10.0 * (pitch**2)

        # 3. Base reward components
        energy_penalty = 0.001 * float(np.sum(np.square(action)))
        fall_penalty = 100.0 if torso_height < self.fall_threshold else 0.0

        return progress_reward + goal_reward + alive_bonus - energy_penalty - fall_penalty - crouch_penalty - symmetry_penalty - ankle_penalty - knee_penalty - torso_tilt_penalty

    # ================================================================== #
    #  Termination / truncation
    # ================================================================== #

    def _is_terminated(self):
        """True when the robot has fallen, reached the goal, or tilted past 20 degrees."""
        torso_height = self._get_torso_height()
        if torso_height < self.fall_threshold:
            return True

        base_pos, base_quat = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )

        if "biped2d" in self.urdf_path:
            torso_state = p.getLinkState(self.robot_id, self.torso_link_id, physicsClientId=self.physics_client)
            base_x = torso_state[0][1]
            torso_quat = torso_state[1]
            torso_euler = p.getEulerFromQuaternion(torso_quat)
            pitch = torso_euler[0]
        else:
            base_x = base_pos[0]
            base_euler = p.getEulerFromQuaternion(base_quat)
            pitch = base_euler[1]

        if base_x >= self.goal_x:
            return True

        if abs(pitch) > 0.35:  # Torso tilt limit of 20 degrees (~0.35 rad)
            return True

        return False

    def _is_truncated(self):
        """True when the episode has exceeded max_episode_steps."""
        return self.step_count >= self.max_episode_steps

    # ================================================================== #
    #  Gymnasium API
    # ================================================================== #

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Apply torques to every actuated joint
        for i, j in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.TORQUE_CONTROL,
                force=float(action[i] * self.max_torque),
                physicsClientId=self.physics_client,
            )

        p.stepSimulation(physicsClientId=self.physics_client)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        info = {
            "step": self.step_count,
            "torso_height": float(self._get_torso_height()),
        }

        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Full simulation reset
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(self.time_step, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Reload assets
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, self.initial_height],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            physicsClientId=self.physics_client,
        )

        # Re-discover joints, torso link, and disable default motors
        self._discover_joints()
        self._discover_torso_link()
        self._disable_default_motors()

        self.step_count = 0
        self.prev_action = np.zeros(self.num_joints, dtype=np.float32)
        self.prev_dist_to_goal = self.goal_x

        return self._get_obs(), {}

    def render(self):
        """PyBullet GUI renders automatically — nothing extra needed."""
        pass

    def close(self):
        """Disconnect from the physics server."""
        if p.isConnected(physicsClientId=self.physics_client):
            p.disconnect(physicsClientId=self.physics_client)
