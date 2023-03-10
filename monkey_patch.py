import os
import pathlib
import re
import warnings

import composer


class WandBLogger(composer.loggers.WandBLogger):
    def upload_file(
        self,
        state: composer.State,
        remote_file_name: str,
        file_path: pathlib.Path,
        *,
        overwrite: bool,
    ):
        del overwrite  # unused

        if not self._enabled or not self._log_artifacts:
            return

        import wandb

        # Some WandB-specific alias extraction
        timestamp = state.timestamp
        aliases = ["latest", f"ep{int(timestamp.epoch)}-ba{int(timestamp.batch)}"]

        # replace all unsupported characters with periods
        # Only alpha-numeric, periods, hyphens, and underscores are supported by wandb.
        new_remote_file_name = re.sub(r"[^a-zA-Z0-9-_\.]", ".", remote_file_name)
        if new_remote_file_name != remote_file_name:
            warnings.warn(
                (
                    "WandB permits only alpha-numeric, periods, hyphens, and underscores in file names. "
                    f"The file with name '{remote_file_name}' will be stored as '{new_remote_file_name}'."
                )
            )

        extension = new_remote_file_name.split(".")[-1]

        if extension == "txt":
            return

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

        # Change the extension so the checkpoint is compatible with W&B's model registry
        if extension == "pt":
            extension = "model"

        wandb_artifact = wandb.Artifact(
            name=new_remote_file_name,
            type=extension,
            metadata=metadata,
        )
        wandb_artifact.add_file(os.path.abspath(file_path))
        wandb.log_artifact(wandb_artifact, aliases=aliases)
