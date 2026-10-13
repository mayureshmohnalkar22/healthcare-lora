import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from healthcare_lora.baseline import TfidfBaseline


def test_json_export_preserves_probabilities(tmp_path):
    texts = [
        "heart cardiac coronary",
        "cancer tumor carcinoma",
        "heart coronary artery",
        "cancer tumor neoplasm",
    ]
    labels = ["methods", "results", "methods", "results"]
    v = TfidfVectorizer().fit(texts)
    # Binary sklearn logistic regression exports one decision row; production export is multiclass.
    # Use five classes here to exercise the actual production format.
    texts += ["gastric bowel intestine", "brain seizure dementia", "pain infection fever"]
    labels += ["objective", "background", "conclusions"]
    v = TfidfVectorizer(ngram_range=(1, 2)).fit(texts)
    c = LogisticRegression().fit(v.transform(texts), labels)
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "vectorizer_params": {"ngram_range": [1, 2]},
                "vocabulary": v.vocabulary_,
                "idf": v.idf_.tolist(),
                "coef": c.coef_.tolist(),
                "intercept": c.intercept_.tolist(),
                "classes": c.classes_.tolist(),
                "manifest": {},
            }
        )
    )
    loaded = TfidfBaseline(path)
    for text in texts:
        r = loaded.predict(text)
        assert r["prediction"] == c.predict(v.transform([text]))[0]
        assert np.isclose(r["confidence"], c.predict_proba(v.transform([text])).max())
