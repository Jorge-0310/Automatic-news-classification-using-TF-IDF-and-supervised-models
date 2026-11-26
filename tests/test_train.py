import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train import CONFIGURATIONS, LABELS, build_pipeline, flatten_row, measure
from error_analysis import select_error_indices


class PipelineTests(unittest.TestCase):
    def test_error_selection_is_unique_and_covers_multiple_pairs(self):
        truth = [0, 0, 0, 1, 1, 2]
        predicted = [1, 1, 1, 0, 0, 0]
        selected = select_error_indices(truth, predicted, 5)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertEqual(len(selected), 5)
        self.assertIn(3, selected[:3])

    def test_each_model_trains_and_reports_all_classes(self):
        x = [
            f"{topic} {topic} {word} {word} report story today"
            for topic, word in (
                ("world", "diplomacy"), ("sports", "tournament"),
                ("business", "markets"), ("science", "technology"),
            )
            for _ in range(4)
        ]
        y = [label for label in LABELS for _ in range(4)]
        for config in CONFIGURATIONS:
            with self.subTest(config=config["id"]):
                model = build_pipeline(config)
                model.fit(x, y)
                self.assertEqual(list(model.classes_), list(LABELS))
                metrics, predictions = measure(model, x, y)
                self.assertEqual(len(predictions), len(x))
                self.assertEqual(len(metrics["per_class"]), 4)
                self.assertGreaterEqual(metrics["f1_macro"], 0)
                row = flatten_row(config, metrics, 0.01, 16, 16, 42)
                self.assertEqual(row["random_state"], 42)
                self.assertIn("Business_recall", row)
                self.assertEqual(model.named_steps["tfidf"].ngram_range,
                                 config["ngram_range"])


if __name__ == "__main__":
    unittest.main()
