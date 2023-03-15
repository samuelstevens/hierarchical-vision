"""
All composer-compatible algorithms.
"""
import dataclasses
import logging
import os
import re

import composer
import torch
from composer.algorithms import (
    EMA,
    BlurPool,
    ChannelsLast,
    GradientClipping,
    ProgressiveResizing,
)

import wandb

log = logging.getLogger(__name__)

__all__ = [
    "BlurPool",
    "CaptureFeatures",
    "ChannelsLast",
    "EMA",
    "GradientClipping",
    "PretrainedBackbone",
    "ProgressiveResizing",
    "LabelSmoothing",
]


class CaptureFeatures(composer.Algorithm):
    def __init__(self, train, eval, save):
        self.train = train
        self.eval = eval
        self.save = os.path.join(save, "features")

    def match(self, event, state):
        if self.train and event == composer.Event.AFTER_FORWARD:
            return True
        if self.eval and event == composer.Event.EVAL_AFTER_FORWARD:
            return True

        return False

    def apply(self, event, state, logger):
        """
        Put the outputs in a big vector. At the end of evaluation,
        save it to disk somewhere.
        """
        breakpoint()


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
            # This actually makes the PretrainedBackbone algorithm run *last*.
            return composer.core.passes.sort_to_front(algorithms, cls)

        return sort

    def match(self, event, state):
        return event == composer.Event.INIT

    def apply(self, event, state, logger):
        self.load_pretrained_backbone(
            self.checkpoint, state.model.module, self.local_cache, self.strict
        )

    @staticmethod
    def load_pretrained_backbone(checkpoint, model_with_backbone, local_cache, strict):
        assert hasattr(model_with_backbone, "backbone")

        downloaded_filepath = (
            wandb.run.use_artifact(checkpoint.url)
            .get_path(checkpoint.filepath)
            .download(root=local_cache)
        )
        state_dict = torch.load(downloaded_filepath, map_location="cpu")["state"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["model"], "module."
        )
        # Delete the linear head keys
        head_keys = {
            key for key in state_dict["model"].keys() if "fc." in key or "head." in key
        }
        for key in head_keys:
            del state_dict["model"][key]

        missing, unexpected = model_with_backbone.backbone.load_state_dict(
            state_dict["model"], strict=strict
        )
        # Don't worry about missing the head keys.
        missing = [key for key in missing if key not in head_keys]

        if missing:
            log.warn(f"Missing keys in checkpoint: {', '.join(missing)}")
        if unexpected:
            log.warn(f"Unexpected keys in checkpoint: {', '.join(unexpected)}")


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


def smooth_labels(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.1):
    """Copied from MosaicML's composer library"""
    target = composer.loss.utils.ensure_targets_one_hot(logits, target)
    n_classes = logits.shape[1]
    return (target * (1.0 - smoothing)) + (smoothing / n_classes)
