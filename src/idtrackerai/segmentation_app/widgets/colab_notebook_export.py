"""The "Save Colab notebook..." button shared by the three model panels.

Detectron2, SAM 3 and YOLO all end the same way for someone without a CUDA
card: the model has to run somewhere else. This writes the session as it
stands in the app, with the panel's own model, into one notebook that needs
only the Drive folder holding the videos and the weights.
"""

from pathlib import Path

from qtpy.QtWidgets import QFileDialog, QMessageBox, QPushButton

from idtrackerai.extra_tools.detectron2_pipeline import colab_notebook


class ColabNotebookExport:
    """For a panel with ``model_spec()``. The app sets ``session_parameters``
    to a callable giving the session's parameters as they would be saved."""

    session_parameters = staticmethod(lambda: None)
    notebook_button: QPushButton

    def make_notebook_button(self) -> QPushButton:
        self.notebook_button = QPushButton("Save Colab notebook...")
        self.notebook_button.clicked.connect(self.choose_colab_notebook)
        return self.notebook_button

    def save_colab_notebook(self, path: Path) -> tuple[str | None, list[str]]:
        """Writes the notebook. Returns what stopped it (or None), and the
        names of the files that have to be on Drive."""
        parameters = self.session_parameters()
        if not parameters or not parameters.get("video_paths"):
            return "Open the session's video first.", []
        spec = self.model_spec()  # type: ignore[attr-defined]
        if spec is None:
            return "Choose the model's weights first.", []
        if not parameters.get("number_of_animals") and not parameters.get(
            "track_wo_identities"
        ):
            return (
                ("Set the number of animals first, or tick Track without"
                " identities: the notebook tracks with what is set here."),
                [],
            )
        session = dict(parameters) | {"segmentation_model": spec}
        if not session.get("name"):
            session["name"] = Path(str(session["video_paths"][0])).stem
        try:
            files = colab_notebook.write_notebook(Path(path), [session])
        except (colab_notebook.ColabNotebookError, OSError) as exc:
            return str(exc), []
        self._notebook_session = session
        return None, files

    def choose_colab_notebook(self) -> None:
        parameters = self.session_parameters() or {}
        name = parameters.get("name") or "session"
        folder = Path(str(parameters.get("output_dir") or Path.cwd()))
        chosen, _ = QFileDialog.getSaveFileName(
            self,  # type: ignore[arg-type]
            "Save the Colab notebook",
            str(folder / f"{name}_colab.ipynb"),
            filter="Notebook (*.ipynb)",
        )
        if not chosen:
            return
        problem, files = self.save_colab_notebook(Path(chosen))
        if problem:
            QMessageBox.warning(self, "Could not write the notebook", problem)  # type: ignore[arg-type]
            return
        QMessageBox.information(
            self,  # type: ignore[arg-type]
            "Colab notebook written",
            f"{Path(chosen).name}\n\n"
            + colab_notebook.upload_message(files, [self._notebook_session]),
        )
