"""Centralized reproducible datasets and classification splits.

Both classification notebooks MUST call load_emotion_splits(SEED), rather than
creating their own splits, to avoid incomparable evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

EMOTION_DATASET_ID = "avsolatorio/mteb-emotion-avs_triplets"
SHAKESPEARE_TRAIN_IDS = ("shakespeare-caesar.txt", "shakespeare-macbeth.txt")
SHAKESPEARE_EVAL_ID = "shakespeare-hamlet.txt"
DEFAULT_SEED = 42


@dataclass(frozen=True)
class LabeledTextSplit:
    texts: list[str]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class EmotionSplits:
    train: LabeledTextSplit
    val: LabeledTextSplit
    test: LabeledTextSplit
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclass(frozen=True)
class ShakespeareCorpora:
    train_documents: tuple[str, str]
    evaluation_document: str


def split_labeled_texts(
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    seed: int = DEFAULT_SEED,
    class_names_by_id: Mapping[int, str] | None = None,
) -> EmotionSplits:
    """Split into 80% train / 10% validation / 10% test, stratified.

    Two fixed-size stratified splits ensure the requested exact overall sizes
    for a 16,000-row dataset, independently of floating-point rounding.
    The split is computed on raw text; downstream vocabularies must be built
    on the training split only.
    """
    texts = list(texts)
    labels_array = np.asarray(labels, dtype=np.int64)
    if len(texts) != len(labels_array):
        raise ValueError("texts and labels must have matching lengths")
    if not texts:
        raise ValueError("Cannot split an empty dataset")

    classes = np.unique(labels_array)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Expected contiguous integer labels beginning at zero")
    if class_names_by_id is None:
        class_names = tuple(str(index) for index in classes)
    else:
        class_names = tuple(class_names_by_id[int(index)] for index in classes)

    n_rows = len(texts)
    n_test = round(n_rows * 0.10)
    n_val = round(n_rows * 0.10)
    if n_test == 0 or n_val == 0 or n_rows - n_test - n_val == 0:
        raise ValueError("Dataset is too small for an 80/10/10 split")

    all_indices = np.arange(n_rows)
    train_val_idx, test_idx = train_test_split(
        all_indices, test_size=n_test, stratify=labels_array,
        random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=n_val,
        stratify=labels_array[train_val_idx], random_state=seed,
    )

    def as_split(indices: np.ndarray) -> LabeledTextSplit:
        return LabeledTextSplit(
            texts=[texts[int(i)] for i in indices],
            labels=labels_array[indices].copy(),
        )

    return EmotionSplits(
        train=as_split(train_idx),
        val=as_split(val_idx),
        test=as_split(test_idx),
        class_names=class_names,
    )


def load_emotion_splits(*, seed: int = DEFAULT_SEED) -> EmotionSplits:
    """Load Hugging Face Emotion dataset and create common 80/10/10 splits.

    Original dataset has one 'train' split of 16,000 labeled sentences.
    'query_idx', 'positive_idx', and 'negative_idx' are not text-classification
    inputs; deliberately ignore them to prevent unintended leakage.
    Requires network access on first run; datasets caches future loads.
    """
    from datasets import load_dataset  # lazily imported to keep helper tests light

    rows = load_dataset(EMOTION_DATASET_ID, split="train")
    required = {"text", "label", "label_text"}
    if not required.issubset(rows.column_names):
        raise ValueError(f"Dataset is missing required columns: {required - set(rows.column_names)}")

    texts = rows["text"]
    labels = rows["label"]
    names = rows["label_text"]

    label_mapping: dict[int, str] = {}
    for label, name in zip(labels, names):
        label, name = int(label), str(name)
        if label in label_mapping and label_mapping[label] != name:
            raise ValueError(f"Inconsistent emotion label name for id {label}")
        label_mapping[label] = name

    if len(label_mapping) != 6:
        raise ValueError(f"Expected 6 emotion classes, got {len(label_mapping)}")

    return split_labeled_texts(
        texts, labels, seed=seed, class_names_by_id=label_mapping,
    )


def load_shakespeare_corpora(*, download_if_missing: bool = True) -> ShakespeareCorpora:
    """Load the precise Gutenberg works specified by Part 1 of the lab."""
    import nltk
    from nltk.corpus import gutenberg

    try:
        gutenberg.fileids()
    except LookupError:
        if not download_if_missing:
            raise
        ok = nltk.download("gutenberg", quiet=True)
        if not ok:
            raise RuntimeError("Could not download NLTK Gutenberg corpus")

    return ShakespeareCorpora(
        train_documents=tuple(gutenberg.raw(fid) for fid in SHAKESPEARE_TRAIN_IDS),
        evaluation_document=gutenberg.raw(SHAKESPEARE_EVAL_ID),
    )
