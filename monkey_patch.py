import os
import pathlib
import re
import warnings

import composer
from torch import Tensor

patched_algorithms = ("LabelSmoothing",)


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


def smooth_labels(logits: Tensor, target: Tensor, smoothing: float = 0.1):
    """Copied from MosaicML's composer library"""
    target = composer.loss.utils.ensure_targets_one_hot(logits, target)
    n_classes = logits.shape[1]
    return (target * (1.0 - smoothing)) + (smoothing / n_classes)


class LabelSmoothing(composer.algorithms.LabelSmoothing):
    """Patched label smoothing that supports hierarchical multitask outputs."""

    def apply(self, event, state, logger):
        if event == composer.Event.BEFORE_LOSS:
            labels = state.batch_get_item(self.target_key)
            assert isinstance(labels, Tensor), "type error"
            self.original_labels = labels.clone()

            if isinstance(state.outputs, list):
                # Multitask outputs.
                # breakpoint()

                # Check shapes match
                b, tiers = labels.shape
                assert len(state.outputs) == tiers, "different level of tiers"
                assert all(
                    output.shape[0] == b for output in state.outputs
                ), "different batch sizes"

                smoothed_labels = []
                for i, (output, tier) in enumerate(zip(state.outputs, labels.T)):
                    smoothed_labels.append(smooth_labels(output, tier, self.smoothing))
                state.batch_set_item(self.target_key, smoothed_labels)
            else:
                # Regular flat labels.
                smoothed_labels = smooth_labels(
                    state.outputs, labels, smoothing=self.smoothing
                )
                state.batch_set_item(self.target_key, smoothed_labels)
        elif event == composer.Event.AFTER_LOSS:
            # restore the target to the non-smoothed version
            state.batch_set_item(self.target_key, self.original_labels)
