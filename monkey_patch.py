import atexit
import os
import pathlib
import re
import warnings

import composer
from composer.utils import dist

import wandb


class WandBLogger(composer.loggers.WandBLogger):
    def init(self, state, logger) -> None:
        """
        Assumes there is already a run initialized on the master process.
        """
        del logger  # unused

        entity_and_project = [None, None]
        if self._enabled:
            assert wandb.run is not None
            entity_and_project = [str(wandb.run.entity), str(wandb.run.project)]
            self.run_dir = wandb.run.dir
            atexit.register(self._set_is_in_atexit)

        # Share the entity and project across all ranks, so they are available on ranks that did not initialize wandb
        dist.broadcast_object_list(entity_and_project)
        self.entity, self.project = entity_and_project
        assert self.entity is not None, "entity should be defined"
        assert self.project is not None, "project should be defined"

    def upload_file(
        self,
        state,
        remote_file_name: str,
        file_path: pathlib.Path,
        *,
        overwrite: bool,
    ):
        del overwrite  # unused

        if not self._enabled or not self._log_artifacts:
            return

        # Some WandB-specific alias extraction
        aliases = [
            "latest",
            f"ep{int(state.timestamp.epoch)}-ba{int(state.timestamp.batch)}",
        ]

        # replace all unsupported characters with periods
        # Only alpha-numeric, periods, hyphens, and underscores are supported by wandb.
        new_remote_file_name = re.sub(r"[^a-zA-Z0-9-_\.]", ".", remote_file_name)
        extension = new_remote_file_name.split(".")[-1]

        if extension in ("txt", "symlink"):
            return

        if extension not in "pt":
            print(extension, "stupid")

        if new_remote_file_name != remote_file_name:
            warnings.warn(
                (
                    "WandB permits only alpha-numeric, periods, hyphens, and underscores in file names. "
                    f"The file with name '{remote_file_name}' will be stored as '{new_remote_file_name}'."
                )
            )

        metadata = {
            f"timestamp/{k}": v for (k, v) in state.timestamp.state_dict().items()
        }
        # if evaluating, also log the evaluation timestamp
        if state.dataloader is not state.train_dataloader:
            # TODO If not actively training, then it is impossible to tell from the state whether
            # the trainer is evaluating or predicting. Assuming evaluation in this case.
            metadata.update(
                {
                    f"eval_timestamp/{k}": v
                    for (k, v) in state.eval_timestamp.state_dict().items()
                }
            )

        wandb_artifact = wandb.Artifact(
            name=new_remote_file_name,
            type="model",
            metadata=metadata,
        )
        wandb_artifact.add_file(os.path.abspath(file_path))
        wandb.log_artifact(wandb_artifact, aliases=aliases)
