import unittest

from spanish_text_classifier.classifier import classify_text


class ClassifierTests(unittest.TestCase):
    def test_classify_text_normalizes_pipeline_output(self):
        def fake_pipeline(text, top_k=None, truncation=None):
            self.assertEqual(text, "Me encanta este producto")
            self.assertEqual(top_k, 3)
            self.assertTrue(truncation)
            return [
                {"label": "POS", "score": 0.91},
                {"label": "NEU", "score": 0.07},
                {"label": "NEG", "score": 0.02},
            ]

        result = classify_text(" Me encanta este producto ", classifier=fake_pipeline)

        self.assertEqual(
            result,
            [
                {"label": "Positivo", "raw_label": "POS", "score": 0.91},
                {"label": "Neutral", "raw_label": "NEU", "score": 0.07},
                {"label": "Negativo", "raw_label": "NEG", "score": 0.02},
            ],
        )

    def test_classify_text_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "Spanish text is required"):
            classify_text("   ", classifier=lambda *_args, **_kwargs: [])


if __name__ == "__main__":
    unittest.main()
