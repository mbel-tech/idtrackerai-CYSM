"""Module to report usage analytics and check for updates."""

import json
import logging
import os
import re
import sys
from contextlib import suppress
from datetime import datetime
from enum import Enum
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from platform import platform, python_version
from threading import Thread
from urllib.request import urlopen

from requests import post

from .py_utils import idtrackerai_version

# A new name on purpose: upstream wrote "true" to usage_analytics_state.json by
# default, so reading that file would carry its opt-out default into this fork.
ANALYTICS_STATE_FILE_PATH = Path(__file__).parent / "usage_analytics_optin.json"
ANALYTICS_URL = "https://analytics.polaviejalab.org/report_usage.php"
PYPI_URL = "https://pypi.org/simple/idtrackerai"
ANALYTICS_ENVIRON = "IDTRACKERAI_DISABLE_ANALYTICS"
ANALYTICS_ENABLE_ENVIRON = "IDTRACKERAI_ENABLE_ANALYTICS"
FORK_URL = "https://github.com/mbel-tech/idtrackerai-CYSM"
UPDATE_HINT = (
    f"To update, get the latest version from {FORK_URL} and reinstall it "
    '(for instance "python -m pip install --upgrade ." from a fresh clone)'
)


class ComparisonResult(Enum):
    EQUAL = "equal"
    MAJOR_UPDATE = "major_update"
    MINOR_UPDATE = "minor_update"
    PATCH_UPDATE = "patch_update"
    STABLE_RELEASE = "stable_release"
    ERROR = "error"

    @property
    def update_available(self) -> bool:
        return self in {
            ComparisonResult.MAJOR_UPDATE,
            ComparisonResult.MINOR_UPDATE,
            ComparisonResult.PATCH_UPDATE,
        }


def check_version_on_console_thread() -> None:
    Thread(target=check_version_on_console).start()


def report_usage_on_console_thread() -> None:
    Thread(target=report_usage).start()


def set_usage_analytics_state(enabled: bool) -> None:
    ANALYTICS_STATE_FILE_PATH.write_text(json.dumps(enabled))


def get_usage_analytics_state() -> bool:
    """Returns whether usage analytics reporting is enabled.

    This fork reports nothing unless the user opts in, either with
    IDTRACKERAI_ENABLE_ANALYTICS=1 or with the "Report usage analytics" switch
    in the apps. IDTRACKERAI_DISABLE_ANALYTICS=1 always wins.
    """
    if os.environ.get(ANALYTICS_ENVIRON, "").lower() in ("1", "true"):
        return False
    if os.environ.get(ANALYTICS_ENABLE_ENVIRON, "").lower() in ("1", "true"):
        return True
    try:
        return json.loads(ANALYTICS_STATE_FILE_PATH.read_text()) is True
    except (OSError, ValueError):
        return False


def report_usage() -> None:
    """Reports usage analytics to the server."""
    usage_analytics_enabled = get_usage_analytics_state()
    if not usage_analytics_enabled:
        logging.info("Usage analytics reporting is disabled")
        return

    try:
        response = post(
            ANALYTICS_URL,
            json={
                "date": datetime.now().astimezone().isoformat(),
                "platform": platform(True),
                "idtrackerai_version": idtrackerai_version(),
                "python_version": python_version(),
                "command": Path(sys.argv[0]).stem if sys.argv else "",  # never paths
            },
        )
        if response.status_code != 200:
            logging.error(
                f"Error reporting usage analytics. Status code: {response.status_code} {response.text}"
            )
    except Exception as e:
        logging.error(f"Error reporting usage analytics: {e}")


def check_version_on_console() -> None:
    with suppress(Exception):
        kind, message = check_version()
        if kind == ComparisonResult.ERROR:
            logging.error(message)
        elif kind != ComparisonResult.EQUAL:
            logging.warning(message)


def parse_release(version: str) -> tuple[int, ...]:
    """The numeric release of a version string, ignoring pre-release and local parts.

    "6.0.15a0+detectron2.1" and "6.0.15+detectron2.1" both give (6, 0, 15).
    """
    release = re.match(r"\s*v?(\d+(?:\.\d+)*)", version)
    if release is None:
        raise ValueError(f"Not a valid version string: {version!r}")
    return tuple(map(int, release.group(1).split(".")))


@lru_cache(maxsize=1)
def check_version() -> tuple[ComparisonResult, str]:
    """Check if there is a new version of idtracker.ai available."""
    try:
        out_text = urlopen(PYPI_URL, timeout=5).read().decode("utf-8")
    except Exception:
        return ComparisonResult.ERROR, "Error fetching PyPI data"

    if not isinstance(out_text, str) or not out_text:
        return ComparisonResult.ERROR, "Error reading PyPI data"

    no_yanked_versions = "\n".join(
        line for line in out_text.splitlines() if "yanked" not in line
    )
    matches: list[tuple[str, str]] = re.findall(
        ">idtrackerai-(.+?)(.tar.gz|-py3-none-any.whl)<", no_yanked_versions
    )

    current_version_str = idtrackerai_version()
    try:
        current_version = parse_release(current_version_str)
    except Exception as e:
        logging.error(f"Error parsing current version: {e}")
        return ComparisonResult.ERROR, (
            f"The current version of idtracker.ai ({current_version}) is not a "
            "valid version string."
        )

    stable_versions = [
        tuple(map(int, version.split(".")))
        for version, _file_extension in matches
        if version.replace(".", "").isdigit()  # only keep stable versions
    ]
    if not stable_versions:
        return ComparisonResult.ERROR, "No stable release found in the PyPI data"
    latest_version = max(stable_versions)

    latest_version_str = ".".join(map(str, latest_version))

    try:
        if latest_version[0] > current_version[0]:
            return ComparisonResult.MAJOR_UPDATE, (
                f"A new major release of idtracker.ai is available: {current_version_str} -> "
                f"{latest_version_str}\n"
                f"{UPDATE_HINT}"
            )
        elif latest_version[1] > current_version[1]:
            return ComparisonResult.MINOR_UPDATE, (
                f"A new minor release of idtracker.ai is available: {current_version_str} -> "
                f"{latest_version_str}\n"
                f"{UPDATE_HINT}"
            )
        elif latest_version[2] > current_version[2]:
            return ComparisonResult.PATCH_UPDATE, (
                f"A new patch release of idtracker.ai is available: {current_version_str} -> "
                f"{latest_version_str}\n"
                f"{UPDATE_HINT}"
            )
        elif re.search(r"(a|b|rc)\d*$", current_version_str.split("+")[0]):
            return ComparisonResult.STABLE_RELEASE, (
                "You are running a pre-release version of idtracker.ai and the stable"
                f" version is available: {current_version_str} ->"
                f" {latest_version_str}\n{UPDATE_HINT}"
            )
        else:
            return ComparisonResult.EQUAL, (
                "There are currently no updates available.\n"
                f"Current idtrackerai version: {current_version_str}"
            )
    except IndexError as exc:
        return ComparisonResult.ERROR, (
            f"Error comparing versions {latest_version_str} and {current_version}: {exc}"
        )
