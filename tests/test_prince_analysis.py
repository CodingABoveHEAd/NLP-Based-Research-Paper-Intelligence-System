import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from prince_analysis import (  # noqa: E402
    cosine_recommendations,
    evaluate_embedding_models,
    normalize_dataset,
    potential_research_gaps,
    trend_table,
)


def sample_data():
    rows = []
    for category, word in [("NLP", "language token"), ("VISION", "image pixel"), ("SECURITY", "network threat")]:
        for year in range(2021, 2025):
            rows.append({"Title": f"{category} paper", "Abstract": f"{word} method", "Category": category, "Year": year})
    return normalize_dataset(pd.DataFrame(rows))


def test_common_split_comparison_reports_both_models():
    data = sample_data()
    rng = np.random.default_rng(42)
    embeddings = {"Word2Vec": rng.normal(size=(len(data), 8)), "BERT": rng.normal(size=(len(data), 12))}
    comparison, results = evaluate_embedding_models(embeddings, data["analysis_category"])
    assert set(comparison["Model"]) == {"Word2Vec", "BERT"}
    assert len(results["Word2Vec"]["test_indices"]) == len(results["BERT"]["test_indices"])
    assert results["Word2Vec"]["confusion_matrix"].shape == (3, 3)


def test_recommendations_exclude_source_and_rank_similar_items():
    data = sample_data()
    embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32)
    recommendations = cosine_recommendations(embeddings, data, source_index=0, count=2)
    assert 0 not in recommendations["Source index"].tolist()
    assert recommendations.iloc[0]["Source index"] == 1


def test_year_trends_and_potential_gaps_use_normalized_year():
    data = sample_data()
    assert len(trend_table(data)) == 12
    assert "Potential combination" in potential_research_gaps(data).columns
