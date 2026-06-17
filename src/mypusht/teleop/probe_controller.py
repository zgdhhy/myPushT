import time

from mypusht.teleop.runtime import configure_runtime

configure_runtime()
import pygame


pygame.init()
pygame.joystick.init()

count = pygame.joystick.get_count()
print("joystick_count:", count)

if count == 0:
    raise SystemExit("No controller detected. Plug in the PS5 controller with USB first.")

joystick = pygame.joystick.Joystick(0)
joystick.init()

print("name:", joystick.get_name())
print("guid:", joystick.get_guid() if hasattr(joystick, "get_guid") else "no guid")
print("num_axes:", joystick.get_numaxes())
print("num_buttons:", joystick.get_numbuttons())
print("num_hats:", joystick.get_numhats())
print()
print("Move sticks and press buttons. Press Ctrl+C to quit.")

try:
    while True:
        pygame.event.pump()
        pygame.event.clear()

        axes = [round(joystick.get_axis(i), 3) for i in range(joystick.get_numaxes())]
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
        hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]

        pressed = [i for i, value in enumerate(buttons) if value]
        print(f"axes={axes} pressed_buttons={pressed} hats={hats}", end="\r")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nquit")
finally:
    pygame.quit()
