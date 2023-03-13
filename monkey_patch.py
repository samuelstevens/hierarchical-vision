import atexit
import dataclasses
import logging
import os
import pathlib
import re
import warnings

import composer
import torch
from composer.utils import dist

import wandb

patched_algorithms = ("LabelSmoothing", "PretrainedBackbone")
log = logging.getLogger(__name__)


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


def smooth_labels(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.1):
    """Copied from MosaicML's composer library"""
    target = composer.loss.utils.ensure_targets_one_hot(logits, target)
    n_classes = logits.shape[1]
    return (target * (1.0 - smoothing)) + (smoothing / n_classes)


class LabelSmoothing(composer.algorithms.LabelSmoothing):
    """Patched label smoothing that supports hierarchical multitask outputs."""

    def apply(self, event, state, logger):
        if event == composer.Event.BEFORE_LOSS:
            labels = state.batch_get_item(self.target_key)
            assert isinstance(labels, torch.Tensor), "type error"
            self.original_labels = labels.clone()

            if isinstance(state.outputs, list):
                # Multitask outputs.

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


@dataclasses.dataclass(frozen=True)
class WandbCheckpoint:
    source: str
    url: str
    filepath: str

    @classmethod
    def parse(cls, uri):
        match = re.match(r"^wandb://([\w./-]+:[\w./-]+)\?([\w./-]+)$", uri)
        if not match:
            raise ValueError(f"uri '{uri}' doesn't match pattern!")

        return cls("wandb", *match.groups())


class PretrainedBackbone(composer.Algorithm):
    """
    Loads a pretrained backbone into the model after WandBLogger and BlurPool algorithms have been initialized.

    This is one the better hacks in my lifetime because it actually doesn't feel that hacky. It feels bad to have to write, but I don't really do anything that isn't supported by composer.
    """

    def __init__(self, checkpoint, local_cache, strict):
        self.checkpoint = WandbCheckpoint.parse(checkpoint)
        self.local_cache = os.path.join(
            local_cache, "wandb-artifacts", self.checkpoint.url
        )
        self.strict = strict

        os.makedirs(self.local_cache, exist_ok=True)

    @classmethod
    def get_algorithm_pass(cls):
        # This ensures that the blurpool algorithm runs first.
        # Then we won't have missing keys.
        def sort(algorithms, event):
            return composer.core.passes.sort_to_back(algorithms, cls)

        return sort

    def match(self, event, state):
        return event == composer.Event.INIT

    def apply(self, event, state, logger):
        missing, unexpected = self._load_pretrained_backbone(state.model.module)
        if missing:
            log.warn(f"Missing keys in checkpoint: {', '.join(missing)}")
        if unexpected:
            log.warn(f"Unexpected keys in checkpoint: {', '.join(unexpected)}")

    def _load_pretrained_backbone(self, model_with_backbone):
        assert hasattr(model_with_backbone, "backbone")

        downloaded_filepath = (
            wandb.run.use_artifact(self.checkpoint.url)
            .get_path(self.checkpoint.filepath)
            .download(root=self.local_cache)
        )
        state_dict = torch.load(downloaded_filepath, map_location="cpu")["state"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["model"], "module."
        )
        missing, unexpected = model_with_backbone.backbone.load_state_dict(
            state_dict["model"], strict=self.strict
        )
        return missing, unexpected
