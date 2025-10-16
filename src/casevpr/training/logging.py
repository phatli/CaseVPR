"""Logging utilities for CaseVPR training."""
from __future__ import annotations

import logging
import os
import sys
import traceback
from typing import Optional


def setup_logging(
    output_folder: str,
    console: Optional[str] = "debug",
    info_filename: Optional[str] = "info.log",
    debug_filename: Optional[str] = "debug.log",
) -> None:
    """Configure root logging handlers.

    Args:
        output_folder: Directory where log files will be stored.
        console: Console verbosity level. ``"debug"`` prints debug and above,
            ``"info"`` prints info and above, ``None`` disables console output.
        info_filename: Name for the info-level log file. If ``None`` the file
            is not created.
        debug_filename: Name for the debug-level log file. If ``None`` the file
            is not created.
    """
    os.makedirs(output_folder, exist_ok=True)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("shapely").disabled = True
    logging.getLogger("shapely.geometry").disabled = True

    base_formatter = logging.Formatter("%(asctime)s   %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)
    logging.getLogger("PIL").setLevel(logging.INFO)

    if info_filename is not None:
        info_file_handler = logging.FileHandler(os.path.join(output_folder, info_filename))
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)

    if debug_filename is not None:
        debug_file_handler = logging.FileHandler(os.path.join(output_folder, debug_filename))
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)

    if console is not None:
        console_handler = logging.StreamHandler()
        if console == "debug":
            console_handler.setLevel(logging.DEBUG)
        elif console == "info":
            console_handler.setLevel(logging.INFO)
        else:
            raise ValueError(f"Unsupported console level: {console}")
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)

    def exception_handler(exc_type, exc_value, exc_traceback):
        logger.info("\n" + "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

    sys.excepthook = exception_handler


def stop_logging() -> None:
    """Detach all handlers from the root logger."""
    logger = logging.getLogger("")
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
