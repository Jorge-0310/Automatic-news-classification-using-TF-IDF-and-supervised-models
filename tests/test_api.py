import json
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import api
from dashboard_data import dashboard_metrics


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)

    def test_frontend_and_metadata(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertIn("NewsScope", self.client.get("/").text)
        health = self.client.get("/api/health").json()
        self.assertIn("model_loaded", health)
        self.assertEqual(health["model_name"], dashboard_metrics()["selected_id"])
        self.assertEqual(self.client.get("/api/model").json()["classes"],
                         ["World", "Sports", "Business", "Sci/Tech"])

    def test_metrics_match_artifacts(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        actual = response.json()
        with (ROOT / "experiments" / "run_config.json").open(encoding="utf-8") as source:
            original = json.load(source)
        self.assertEqual(actual["test"]["f1_macro"], original["test"]["f1_macro"])
        self.assertEqual(actual["test"]["accuracy"], original["test"]["accuracy"])
        self.assertEqual(actual["confusion_matrix"], original["test_confusion_matrix"])
        self.assertEqual(len(actual["comparison"]), 4)
        with (ROOT / "experiments" / "dashboard_metrics.json").open(encoding="utf-8") as source:
            self.assertEqual(actual, json.load(source))
        cost = original["final_computational_cost"]
        self.assertEqual(actual["final_computational_cost"]["energy_kwh"], cost["energy_kwh"])
        self.assertEqual(
            actual["final_computational_cost"]["emissions_kg_co2eq"],
            cost["emissions_kg_co2eq"],
        )
        with (ROOT / "experiments" / "model_comparison.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            comparison = list(csv.DictReader(source))
        self.assertEqual(
            actual["comparison"][1]["energy_kwh"],
            float(comparison[1]["energy_kwh"]),
        )
        self.assertNotIn("carbon_file", response.text)
        self.assertNotIn("carbon_measurement_file", response.text)
        self.assertNotIn('"text":', response.text)

    def test_errors_are_aggregated_without_corpus(self):
        response = self.client.get("/api/errors/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertNotIn("cases", data)
        self.assertNotIn("test_index", response.text)
        self.assertNotIn("cause_hypothesis", response.text)
        self.assertNotIn('"text":', response.text)
        if data["status"] == "confirmed":
            self.assertEqual(sum(pair["count"] for pair in data["pairs"]), data["count"])
            with (ROOT / "experiments" / "error_review_confirmed.json").open(encoding="utf-8") as source:
                review = json.load(source)
            for case in review["cases"]:
                self.assertNotIn(case["cause_hypothesis"], response.text)
                self.assertNotIn(case["limitation"], response.text)

    def test_validation_and_unavailable_model(self):
        self.assertEqual(self.client.post("/api/predict", json={"text": "  "}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict", json={"text": "x" * 5001}).status_code, 413)
        malformed = self.client.post("/api/predict", json={"text": {"private": "SECRET"}})
        self.assertEqual(malformed.status_code, 422)
        self.assertNotIn("SECRET", malformed.text)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(api, "MODEL_PATH", Path(folder) / "absent.joblib"):
                response = self.client.post("/api/predict", json={"text": "A headline"})
                self.assertEqual(response.status_code, 503)

    def test_prediction_contract_without_loading_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "local.joblib"
            marker.touch()
            with patch.object(api, "MODEL_PATH", marker), patch.object(api, "predict", return_value={
                "predicted_class": "Sports",
                "probabilities": {"World": 0.01, "Sports": 0.97,
                                  "Business": 0.01, "Sci/Tech": 0.01},
            }):
                result = self.client.post("/api/predict", json={"text": "The team won"}).json()
        self.assertEqual(result["predicted_class"], "Sports")
        self.assertEqual(len(result["probabilities"]), 4)
        self.assertNotIn("The team won", str(result))


if __name__ == "__main__":
    unittest.main()
