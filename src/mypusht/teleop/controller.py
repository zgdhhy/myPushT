from mypusht.teleop.runtime import configure_runtime

configure_runtime()
import pygame

from mypusht.teleop.config import ControllerMapping


class ButtonLatch:
    def __init__(self) -> None:
        self._last_state: dict[int, int] = {}

    def pressed_once(self, joystick: pygame.joystick.Joystick, button_id: int) -> bool:
        current = joystick.get_button(button_id)
        rising = current == 1 and self._last_state.get(button_id, 0) == 0
        self._last_state[button_id] = current
        return rising


def setup_controller() -> pygame.joystick.Joystick:
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count == 0:
        raise RuntimeError("No PS5/controller detected by pygame.")

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("controller:", joystick.get_name())
    return joystick


def read_stick_delta(joystick: pygame.joystick.Joystick, mapping: ControllerMapping) -> tuple[float, float]:
    ax = joystick.get_axis(mapping.axis_left_x)
    ay = joystick.get_axis(mapping.axis_left_y)

    dx = ax if abs(ax) > mapping.deadzone else 0.0
    dy = -ay if abs(ay) > mapping.deadzone else 0.0
    return dx, dy
