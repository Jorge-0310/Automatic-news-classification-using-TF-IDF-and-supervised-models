import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import review_errors


class HumanReviewTests(unittest.TestCase):
    def test_export_omits_source_text_and_requires_ten_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory / "errors.csv"
            fields = ("test_index", "text", "true_class", "predicted_class",
                      "predicted_probability", "error_type", "cause_hypothesis", "limitation")
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for index in range(10):
                    writer.writerow({
                        "test_index": index, "text": "PRIVATE ARTICLE TEXT",
                        "true_class": "World", "predicted_class": "Business",
                        "predicted_probability": 0.6, "error_type": "Temas compartidos",
                        "cause_hypothesis": "Vocabulario mixto",
                        "limitation": "Una etiqueta por noticia",
                    })
            with patch.object(review_errors, "EXPERIMENTS", directory):
                with patch("builtins.input", side_effect=["Investigador"] + ["s"] * 10):
                    with redirect_stdout(io.StringIO()):
                        review_errors.main()
                self.assertFalse((directory / "error_review_confirmed.json").exists())
                with patch("builtins.input", side_effect=["Investigador"] + ["a"] * 10):
                    with redirect_stdout(io.StringIO()):
                        review_errors.main()
            result = (directory / "error_review_confirmed.json").read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE ARTICLE TEXT", result)
            self.assertEqual(json.loads(result)["count"], 10)


if __name__ == "__main__":
    unittest.main()
