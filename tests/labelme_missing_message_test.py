"""The Detectron2 panel's message when LabelMe is not installed.

LabelMe 7 needs Python 3.12 or later, so on 3.10 and 3.11 pip cannot install
it, and the message must not suggest a pip command that installs nothing.
"""

import pytest

pytest.importorskip("qtpy", reason="the Segmentation App needs Qt")

from idtrackerai.segmentation_app.widgets.detectron2_panel import (  # noqa: E402
    labelme_missing_message,
)


@pytest.mark.parametrize("python", [(3, 10, 4), (3, 11, 9)])
def test_older_python_is_told_labelme_cannot_be_installed(python):
    message = labelme_missing_message(python, "/env/bin/python")
    assert "Python 3.12" in message
    assert f"Python {python[0]}.{python[1]}" in message
    assert "pip install" not in message


@pytest.mark.parametrize("python", [(3, 12, 0), (3, 13, 2)])
def test_supported_python_gets_the_install_command(python):
    message = labelme_missing_message(python, "/env/bin/python")
    assert '"/env/bin/python" -m pip install "idtrackerai-CYSM[annotate]"' in message
