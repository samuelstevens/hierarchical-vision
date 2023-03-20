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

import swinv2
import wandb

log = logging.getLogger(__name__)

__all__ = [
    "BlurPool",
    "ChannelsLast",
    "EMA",
    "GradientClipping",
    "PretrainedBackbone",
    "ProgressiveResizing",
    "LabelSmoothing",
]


class PretrainedBackbone(composer.Algorithm):
    """
    Loads a pretrained backbone into the model after WandBLogger and BlurPool algorithms have been initialized.

    This is one the better hacks in my lifetime because it actually doesn't feel that hacky. It feels bad to have to write, but I don't really do anything that isn't supported by composer.
    """

    def __init__(self, checkpoint, local_cache, strict):
        self.checkpoint = WandbCheckpoint.parse(checkpoint)
        self.local_cache = local_cache
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
        model_dict = checkpoint.load_model_dict(local_cache)

        # Delete the linear head keys
        head_keys = {key for key in model_dict.keys() if "fc." in key or "head." in key}
        for key in head_keys:
            del model_dict[key]

        missing, unexpected = model_with_backbone.backbone.load_state_dict(
            model_dict, strict=strict
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
            raise ValueError(f"uri '{uri}' doesn't match the pattern!")

        return cls("wandb", *match.groups())

    def load_model_dict(self, cache):
        cache = os.path.join(cache, "wandb-artifacts", self.url)
        downloaded = (
            wandb.run.use_artifact(self.url)
            .get_path(self.filepath)
            .download(root=cache)
        )
        model_dict = torch.load(downloaded, map_location="cpu")["state"]["model"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            model_dict, "module."
        )
        return model_dict


def parse_checkpoint(uri: str):
    for cls in [WandbCheckpoint, swinv2.Checkpoint]:
        try:
            return cls.parse(uri)
        except ValueError:
            pass

    raise ValueError(f"Could not parse {uri}")


def smooth_labels(logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.1):
    """Copied from MosaicML's composer library"""
    target = composer.loss.utils.ensure_targets_one_hot(logits, target)
    n_classes = logits.shape[1]
    return (target * (1.0 - smoothing)) + (smoothing / n_classes)
