"""
Interactive URDF Viewer — tinker with the biped without any RL.

Opens PyBullet GUI with a debug slider for every actuated joint so you
can drag them in real time and watch how the robot reacts to gravity,
collisions, and joint limits.

Usage
-----
    python viewer.py                                         # default URDF
    python viewer.py --urdf humanoid_leg_12dof.8.urdf        # your own URDF
    python viewer.py --fixed                                 # pin the base in the air
    python viewer.py --no-gravity                            # zero gravity sandbox

PyBullet GUI Controls
---------------------
    Mouse drag          : rotate camera
    Scroll wheel        : zoom
    Ctrl + drag         : pan
    Joint sliders       : appear on the right panel — drag to move joints
    W / S               : move camera forward / back
    A / D               : move camera left / right
"""

import argparse
import time
import pybullet as p
import pybullet_data


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive PyBullet URDF Viewer")
    parser.add_argument(
        "--urdf",
        type=str,
        default="biped/biped2d_pybullet.urdf",
        help="URDF file to load (searched in pybullet_data and working dir)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=1.0,
        help="Initial spawn height of the robot (default: 1.0m)",
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Fix the robot base in the air (useful for inspecting joints)",
    )
    parser.add_argument(
        "--no-gravity",
        action="store_true",
        help="Disable gravity (zero-G sandbox)",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Run at real-time speed instead of as-fast-as-possible",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Connect to PyBullet GUI ---- #
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Physics
    time_step = 1.0 / 240.0
    p.setTimeStep(time_step, physicsClientId=physics_client)

    if args.no_gravity:
        p.setGravity(0, 0, 0, physicsClientId=physics_client)
        print("  Gravity: OFF")
    else:
        p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
        print("  Gravity: ON  (-9.81 m/s²)")

    # ---- Load ground plane ---- #
    p.loadURDF("plane.urdf", physicsClientId=physics_client)

    # ---- Load robot ---- #
    robot_id = p.loadURDF(
        args.urdf,
        basePosition=[0, 0, args.height],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=args.fixed,
        physicsClientId=physics_client,
    )

    # ---- Set up camera ---- #
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=30,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, args.height * 0.5],
        physicsClientId=physics_client,
    )

    # ---- Discover joints and create sliders ---- #
    joint_ids = []
    slider_ids = []
    joint_info_list = []

    num_joints = p.getNumJoints(robot_id, physicsClientId=physics_client)

    print(f"\n  URDF          : {args.urdf}")
    print(f"  Total joints  : {num_joints}")
    print(f"  Fixed base    : {args.fixed}")
    print(f"  Spawn height  : {args.height}m")
    print()

    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i, physicsClientId=physics_client)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]

        # Only create sliders for revolute and prismatic joints
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            lower_limit = info[8]   # joint lower limit
            upper_limit = info[9]   # joint upper limit

            # If limits are both 0, use sensible defaults
            if lower_limit >= upper_limit:
                if joint_type == p.JOINT_REVOLUTE:
                    lower_limit = -3.14
                    upper_limit = 3.14
                else:
                    lower_limit = -1.0
                    upper_limit = 1.0

            # Get current joint position as initial slider value
            current_pos = p.getJointState(
                robot_id, i, physicsClientId=physics_client
            )[0]

            slider_id = p.addUserDebugParameter(
                paramName=f"J{i}: {joint_name}",
                rangeMin=float(lower_limit),
                rangeMax=float(upper_limit),
                startValue=float(current_pos),
                physicsClientId=physics_client,
            )

            joint_ids.append(i)
            slider_ids.append(slider_id)
            joint_info_list.append((i, joint_name, lower_limit, upper_limit))

            type_name = "revolute" if joint_type == p.JOINT_REVOLUTE else "prismatic"
            print(
                f"  [{i:2d}] {joint_name:30s}  {type_name:10s}  "
                f"range: [{lower_limit:.2f}, {upper_limit:.2f}]"
            )

    print(f"\n  Actuated joints with sliders: {len(joint_ids)}")
    print()

    # ---- Create camera target sliders ---- #
    p.addUserDebugParameter("------------------", 1, 0, 1, physicsClientId=physics_client) # divider
    cam_x_slider = p.addUserDebugParameter("Cam Target X", -5.0, 5.0, 0.0, physicsClientId=physics_client)
    cam_y_slider = p.addUserDebugParameter("Cam Target Y", -5.0, 5.0, 0.0, physicsClientId=physics_client)
    cam_z_slider = p.addUserDebugParameter("Cam Target Z", 0.0, 5.0, float(args.height * 0.5), physicsClientId=physics_client)

    prev_cam_x = 0.0
    prev_cam_y = 0.0
    prev_cam_z = float(args.height * 0.5)

    print("=" * 60)
    print("  Drag the sliders on the right panel to move joints.")
    print("  Use the mouse to orbit (click/drag), zoom (scroll), and pan (Ctrl+drag).")
    print("  Alternatively, use the 'Cam Target' sliders to pan/focus the camera.")
    print("  Press Ctrl+C in the terminal to quit.")
    print("=" * 60)
    print()

    # ---- Main loop: read sliders → set joint positions ---- #
    try:
        while True:
            # Update robot joint positions from sliders
            for idx, joint_id in enumerate(joint_ids):
                target = p.readUserDebugParameter(
                    slider_ids[idx], physicsClientId=physics_client
                )
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=50.0,
                    physicsClientId=physics_client,
                )

            # Update camera target position from sliders (preserving yaw, pitch, zoom)
            cam_info = p.getDebugVisualizerCamera(physicsClientId=physics_client)
            if cam_info is not None and len(cam_info) >= 11:
                c_yaw = cam_info[8]
                c_pitch = cam_info[9]
                c_dist = cam_info[10]

                cam_x = p.readUserDebugParameter(cam_x_slider, physicsClientId=physics_client)
                cam_y = p.readUserDebugParameter(cam_y_slider, physicsClientId=physics_client)
                cam_z = p.readUserDebugParameter(cam_z_slider, physicsClientId=physics_client)

                if cam_x != prev_cam_x or cam_y != prev_cam_y or cam_z != prev_cam_z:
                    p.resetDebugVisualizerCamera(
                        cameraDistance=c_dist,
                        cameraYaw=c_yaw,
                        cameraPitch=c_pitch,
                        cameraTargetPosition=[cam_x, cam_y, cam_z],
                        physicsClientId=physics_client
                    )
                    prev_cam_x, prev_cam_y, prev_cam_z = cam_x, cam_y, cam_z

            p.stepSimulation(physicsClientId=physics_client)

            if args.slow:
                time.sleep(time_step)

    except KeyboardInterrupt:
        print("\n  Viewer closed.")

    finally:
        p.disconnect(physicsClientId=physics_client)


if __name__ == "__main__":
    main()
