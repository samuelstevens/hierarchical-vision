import collections
import dataclasses
import os

import composer.metrics
import einops
import torch.nn
import torchmetrics
import torchvision.datasets


class MultitaskHead(torch.nn.Module):
    """
    Hierarchical multitask head

    Adds a linear layer for each "tier" in the hierarchy.

    forward() returns a list of logits for each tier.

    Arguments:
        num_features (int): number of features from the backbone
        num_classes (tuple[int, ...]): a tuple of each number of classes in the hierarchy.
    """

    def __init__(self, num_features, num_classes):
        super().__init__()

        self.num_classes = tuple(num_classes)
        for num_class in self.num_classes:
            assert num_class > 0

        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(num_features, num_class) for num_class in self.num_classes]
        )

    def forward(self, x):
        # we do not want to use self.heads(x) because that would feed them through
        # each element in the list sequentially, whereas we want x through each head
        # individually.
        return [head(x) for head in self.heads]


def multitask_surgery(model, head: str, num_classes):
    """
    Replaces the head with a MultitaskHead.
    """
    if not hasattr(model, head):
        raise RuntimeError(f"model has no attribute {head}!")

    # We use max because we know the number of classes is 2 (from models.py).
    # So we can pick the bigger one because all the models we're using will
    # always have more than 2 features.
    num_features = max(getattr(model, head).weight.shape)

    setattr(model, head, MultitaskHead(num_features, num_classes))


class MultitaskCrossEntropy(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, coeffs=(1.0,), **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(coeffs, torch.Tensor):
            coeffs = coeffs.clone().detach().type(torch.float)
        else:
            coeffs = torch.tensor(coeffs, dtype=torch.float)

        self.register_buffer("coeffs", coeffs)

    def forward(self, inputs, targets):
        if not isinstance(targets, list):
            # If we don't use label smoothing, targets is a
            # B x tiers tensor (2048 x 7 for ResNet + iNat21).
            targets = einops.rearrange(targets, "batch tiers -> tiers batch")

        assert (
            len(inputs) == len(targets) == len(self.coeffs)
        ), f"{len(inputs)} != {len(targets)} != {len(self.coeffs)}"

        losses = torch.stack(
            [
                # Need to specify arguments to super() because of some a bug
                # with super() in list comprehensions/generators (unclear)
                super(MultitaskCrossEntropy, self).forward(input, target)
                for input, target in zip(inputs, targets)
            ]
        )
        return torch.dot(self.coeffs, losses)


class FineGrainedAccuracy(torchmetrics.Metric):
    is_differentiable = False
    higher_is_better = True
    # Try turning this off and see if it improves performance while producing the same results.
    # https://torchmetrics.readthedocs.io/en/stable/pages/implement.html#internal-implementation-details
    full_state_update = True

    def __init__(self, topk=1):
        super().__init__()
        self.topk = topk
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: list[torch.Tensor], target: torch.Tensor):
        assert isinstance(outputs, list)
        assert target.ndim > 1

        preds = fine_grained_predictions(outputs, topk=self.topk).squeeze()
        target = target[:, -1].squeeze()

        breakpoint()
        # Look at the code in accuracy() in src/hierarchical.py in swin-transformer

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class FineGrainedCrossEntropy(composer.metrics.CrossEntropy):
    """
    A cross-entropy used with hierarchical inputs and targets and only
    looks at the finest-grained tier (the last level).
    """

    def update(self, preds: list[torch.Tensor], targets: torch.Tensor):
        if not isinstance(preds, list):
            raise RuntimeError("FineGrainedCrossEntropy needs a list of predictions")
        preds, targets = preds[-1], targets[:, -1]
        super().update(preds, targets)


class HierarchicalCrossEntropy(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()


class HierarchicalImageFolder(torchvision.datasets.ImageFolder):
    """
    Parses an image folder where the hierarchy is represented as follows:

    00000_top_middle_..._bottom
    00001_top_middle_..._other
    ...
    """

    num_classes = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_classes(self, directory):
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )

        tier_lookup = {}
        class_to_idxs = {}

        for cls in classes:
            tiers = HierarchicalLabel.parse(cls).cleaned

            for tier, value in enumerate(tiers):
                if tier not in tier_lookup:
                    tier_lookup[tier] = {}

                if value not in tier_lookup[tier]:
                    tier_lookup[tier][value] = len(tier_lookup[tier])

            class_to_idxs[cls] = torch.tensor(
                [tier_lookup[tier][value] for tier, value in enumerate(tiers)]
            )

        # Set self.num_classes
        self.num_classes = tuple(len(tier) for tier in tier_lookup.values())

        return classes, class_to_idxs


@dataclasses.dataclass(frozen=True)
class HierarchicalLabel:
    raw: str
    number: int
    kingdom: str
    phylum: str
    cls: str
    order: str
    family: str
    genus: str
    species: str

    @classmethod
    def parse(cls, name):
        """
        Sometimes the tree is not really a tree. For example, sometimes there are repeated orders.
        This function fixes that by repeating the upper tier names in the lower tier names.

        Suppose we only had order-level classification. This would be the class for a bald eagle's
        order:

        00001_animalia_chordata_aves_accipitriformes

        Suppose then that we had a another class:

        00002_animalia_chordata_reptilia_accipitriformes

        These accipitriformes refer to different nodes in the tree. To fix this, we do:

        00001_animalia_chordata_aves_accipitriformes ->
            00001, animalia, animalia-chordata, animalia-chordata-aves, animalia-chordata-aves-accipitriformes

        00002_animalia_chordata_reptilia_accipitriformes ->
            00002, animalia, animalia-chordata, animalia-chordata-reptilia, animalia-chordata-reptilia-accipitriformes

        Now each bit of text refers to the same nodes.
        It's not pretty but it does get the job done.

        Arguments:
            name (str): the complete taxonomic name, separated by '_'
        """

        # index is a number
        # top is kingdom
        index, top, *tiers = name.split("_")
        number = int(index)

        cleaned = [top]

        complete = top
        for tier in tiers:
            complete += f"-{tier}"
            cleaned.append(complete)

        assert len(cleaned) == 7, f"{len(cleaned)} != 7"

        return cls(name, number, *cleaned)

    @property
    def cleaned(self):
        return "_".join(
            [
                str(self.number).rjust(5, 0),
                self.kingdom,
                self.phylum,
                self.cls,
                self.order,
                self.family,
                self.genus,
                self.species,
            ]
        )


class LeafCountLookup:
    def __init__(self, labels: list[HierarchicalLabel]):
        self._lookup = collections.defaultdict(int)
        for label in labels:
            self._lookup[(label.kingdom, "kingdom")] += 1
            self._lookup[(label.phylum, "phylum")] += 1
            self._lookup[(label.cls, "cls")] += 1
            self._lookup[(label.order, "order")] += 1
            self._lookup[(label.family, "family")] += 1
            self._lookup[(label.genus, "genus")] += 1
            self._lookup[(label.species, "species")] += 1
        self.total = len(labels)

    def closest(self, n: int | float) -> tuple[str, str, int]:
        """
        Return the label and the level that has the closest count.
        If n is a float, find the count that is closest to n * self.total.
            (the float must be between 0 and 1)

        If n is an int, find the count that is closest n.
        """
        if isinstance(n, float):
            assert 0 <= n <= 1, "n must be fractional"
            n = int(self.total * n)
        assert isinstance(n, int)

        closest, dist = None, float("inf")

        for label, count in self._lookup.items():
            if abs(count - n) < dist:
                closest, dist = (*label, count), abs(count - n)

        if closest is None:
            raise RuntimeError("no values in lookup!")

        return closest


def fine_grained_predictions(output, topk=1, hierarchy_level=-1):
    """
    Computes the top k predictions for a hierarchical output

    Copied from rwightman/pytorch-image-models/timm/utils/metrics.py and modified
    to work with hierarchical outputs as well.

    When the output is hierarchical, only returns the accuracy for `hierarchy_level`
    (default -1, which is the fine-grained level).
    """
    if isinstance(output, list):
        len(output)
        output = output[hierarchy_level]

    batch_size, num_classes = output.shape

    maxk = min(topk, num_classes)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    return pred
