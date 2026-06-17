import os
import warnings


def configure_runtime() -> None:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    warnings.filterwarnings(
        "ignore",
        message="Your system is avx2 capable but pygame was not built with support for it.*",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
