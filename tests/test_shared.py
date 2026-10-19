import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys

import numpy as np

from utils.data_loader import load_emotion_splits, split_labeled_texts
from utils.evaluation import sklearn_evaluation
from utils.preprocessing import normalize_text, tokenize


class SharedInfrastructureTests(unittest.TestCase):
    def test_preprocessing(self):
        self.assertEqual(tokenize("Hello, WORLD! Don't stop. 123"),
                         ["hello", "world", "don", "t", "stop", "123"])
        self.assertEqual(normalize_text("Red--BLUE!"), "red blue")

    def test_split_80_10_10_and_reproducible(self):
        texts = [f"unique-example-{i}" for i in range(160)]
        labels = [i % 4 for i in range(160)]
        a = split_labeled_texts(texts, labels, seed=42)
        b = split_labeled_texts(texts, labels, seed=42)
        self.assertEqual((len(a.train), len(a.val), len(a.test)), (128, 16, 16))
        self.assertEqual(a.train.texts, b.train.texts)
        self.assertEqual(a.val.texts, b.val.texts)
        self.assertEqual(a.test.texts, b.test.texts)
        self.assertEqual(set(a.train.texts) & set(a.val.texts), set())
        self.assertEqual(set(a.train.texts) & set(a.test.texts), set())
        self.assertEqual(set(a.val.texts) & set(a.test.texts), set())
        for part in (a.train, a.val, a.test):
            self.assertTrue(all(np.bincount(part.labels, minlength=4) == len(part) // 4))


    def test_dataset_loader_uses_only_labeled_text_and_fixed_split(self):
        class FakeDataset:
            column_names = ["text", "label", "label_text", "query_idx"]
            def __getitem__(self, key):
                if key == "text":
                    return [f"sentence {i}" for i in range(16000)]
                if key == "label":
                    return [i % 6 for i in range(16000)]
                if key == "label_text":
                    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
                    return [names[i % 6] for i in range(16000)]
                raise KeyError(key)

        fake_module = SimpleNamespace(load_dataset=lambda dataset_id, split: FakeDataset())
        with patch.dict(sys.modules, {"datasets": fake_module}):
            splits = load_emotion_splits(seed=42)
        self.assertEqual((len(splits.train), len(splits.val), len(splits.test)),
                         (12800, 1600, 1600))
        self.assertEqual(splits.class_names[0], "sadness")
        self.assertEqual(splits.class_names[5], "surprise")
        self.assertEqual(splits.num_classes, 6)

    def test_reference_metrics(self):
        report = sklearn_evaluation(
            np.array([0, 1, 1, 2]), np.array([0, 1, 0, 2]), num_classes=3
        )
        self.assertEqual(report["confusion_matrix"].shape, (3, 3))
        self.assertTrue(0 <= report["macro_f1"] <= 1)


if __name__ == "__main__":
    unittest.main()
