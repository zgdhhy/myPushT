import argparse
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, default=None)
    parser.add_argument("--max_seconds", type=float, default=60.0)
    parser.add_argument("--preview_scale", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from mypusht.data.episode_io import next_episode_path, save_episode
    from mypusht.envs.so100_teleop_env import So100TeleopEnv
    from mypusht.teleop.config import load_mapping
    from mypusht.teleop.controller import ButtonLatch, read_stick_delta, setup_controller
    from mypusht.teleop.preview import show_preview
    from mypusht.teleop.runtime import configure_runtime

    configure_runtime()
    import cv2
    import pygame
    
    mapping = load_mapping()
    sample_dt = 1.0 / mapping.fps

    joystick = setup_controller()
    buttons = ButtonLatch()
    env = So100TeleopEnv(
        move_speed=mapping.move_speed,
        workspace_x=mapping.workspace_x,
        workspace_y=mapping.workspace_y,
        image_observation=False,
    )
    env.reset()

    recording = False
    buffer: list[dict] = []
    last_sample_time = 0.0
    start_time = time.time()

    print("Controls from controller_mapping.json")
    print("record button:", mapping.button_record)
    print("reset button:", mapping.button_reset)
    print("exit button:", mapping.button_exit)
    print("discard button:", mapping.button_discard)
    print("Press record once to start, press it again to save the episode.")

    try:
        while True:
            loop_time = time.time()
            elapsed = loop_time - start_time

            pygame.event.pump()
            pygame.event.clear()
            if buttons.pressed_once(joystick, mapping.button_exit):
                print("exit requested")
                break

            if buttons.pressed_once(joystick, mapping.button_reset):
                print("reset")
                recording = False
                buffer = []
                env.reset()

            if buttons.pressed_once(joystick, mapping.button_discard):
                if recording:
                    print("recording discarded")
                    recording = False
                    buffer = []

            if buttons.pressed_once(joystick, mapping.button_record):
                if recording:
                    recording = False
                    save_episode(next_episode_path(), buffer, mapping.fps)
                    buffer = []
                else:
                    print("recording started")
                    recording = True
                    buffer = []
                    start_time = time.time()

            dx, dy = read_stick_delta(joystick, mapping)
            obs, _, _, _, info = env.step([dx, dy])
            step_info = env.step_info(info)

            now = time.time()
            if now - last_sample_time >= sample_dt:
                last_sample_time = now
                cam_top = env.render_camera("top_view")
                cam_side = env.render_camera("side_view")
                mocap_xy = obs["mocap_xy"]

                if recording:
                    buffer.append({
                        "cam_top": cam_top,
                        "cam_side": cam_side,
                        "state": obs["robot_state"],
                        "mocap_xy": obs["mocap_xy"],
                        "object_pose": obs["object_pose"],
                        "goal_pose": obs["goal_pose"],
                        "action": mocap_xy,
                        "success": step_info.success,
                        "xy_error": step_info.xy_error,
                        "yaw_error": step_info.yaw_error,
                        "timestamp": float(now - start_time),
                    })

                show_preview(
                    cam_top=cam_top,
                    cam_side=cam_side,
                    info=step_info,
                    mocap_xy=mocap_xy,
                    recording=recording,
                    frame_count=len(buffer),
                    preview_scale=args.preview_scale
                )

            if recording and elapsed > args.max_seconds:
                print("max_seconds reached, saving")
                recording = False
                save_episode(next_episode_path(), buffer, mapping.fps)
                buffer = []

            sleep_time = env.model.opt.timestep - (time.time() - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if recording and buffer:
            save_episode(next_episode_path(), buffer, mapping.fps)
        cv2.destroyAllWindows()
        pygame.quit()
        env.close()


if __name__ == "__main__":
    main()
