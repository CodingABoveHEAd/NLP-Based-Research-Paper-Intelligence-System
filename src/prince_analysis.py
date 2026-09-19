"""Prince's classification, trend, and potential research-gap pipeline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
CATEGORY_COLUMNS = ("Category", "category", "Topic", "Domain")
YEAR_COLUMNS = ("year", "Year", "Publication year", "Publication Year", "publication_year")


def normalize_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize source columns while preserving every original column."""
    output = dataframe.copy()
    text_column = next((column for column in ("text", "clean_text", "Text") if column in output), None)
    if text_column:
        output["analysis_text"] = output[text_column].fillna("").astype(str).str.strip()
    else:
        title = output.get("Title", pd.Series("", index=output.index)).fillna("").astype(str)
        abstract = output.get("Abstract", pd.Series("", index=output.index)).fillna("").astype(str)
        output["analysis_text"] = (title + " " + abstract).str.strip()
    category_column = next((column for column in CATEGORY_COLUMNS if column in output), None)
    output["analysis_category"] = (
        output[category_column].fillna("").astype(str).str.strip() if category_column else "Unlabelled"
    )
    output["analysis_category"] = output["analysis_category"].replace("", "Unlabelled")
    year_column = next((column for column in YEAR_COLUMNS if column in output), None)
    output["analysis_year"] = (
        pd.to_numeric(output[year_column], errors="coerce").astype("Int64")
        if year_column else pd.Series(pd.NA, index=output.index, dtype="Int64")
    )
    return output[output["analysis_text"].ne("")].reset_index(drop=True)


def load_embedding(path: str | Path, expected_rows: int) -> np.ndarray:
    """Load and validate a saved document embedding matrix."""
    vectors = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if vectors.shape[0] != expected_rows:
        raise ValueError(f"Embedding rows ({vectors.shape[0]}) do not match dataset rows ({expected_rows}).")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix contains NaN or infinite values.")
    return vectors


def evaluate_logistic_regression(embeddings: np.ndarray, labels, seed: int = 42) -> dict:
    """Train and evaluate one multiclass Logistic Regression model."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or pd.Series(labels).value_counts().min() < 2:
        raise ValueError("Every category needs at least two papers and at least two categories.")
    train_x, test_x, train_y, test_y = train_test_split(
        embeddings, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    classes = np.unique(labels)
    report = classification_report(test_y, predictions, output_dict=True, zero_division=0)
    metrics = pd.DataFrame([
        {"Category": label, **values} for label, values in report.items() if isinstance(values, dict)
    ])
    matrix = pd.DataFrame(confusion_matrix(test_y, predictions, labels=classes), index=classes, columns=classes)
    return {
        "model": model,
        "accuracy": accuracy_score(test_y, predictions),
        "metrics": metrics,
        "confusion_matrix": matrix,
        "test_labels": test_y,
        "predictions": predictions,
        "train_count": len(train_y),
        "test_count": len(test_y),
    }


def common_split(labels, test_size: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return one stratified train/test index split reusable by every embedding model."""
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    return train_indices, test_indices


def evaluate_embedding_models(
    embeddings_by_model: dict[str, np.ndarray], labels, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Compare embedding classifiers using exactly the same train/test indices."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or pd.Series(labels).value_counts().min() < 2:
        raise ValueError("Every category needs at least two papers and at least two categories.")
    train_indices, test_indices = common_split(labels, test_size, seed)
    results = {}
    rows = []
    for name, embeddings in embeddings_by_model.items():
        if len(embeddings) != len(labels):
            raise ValueError(f"{name} embedding rows do not match the dataset rows.")
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        model.fit(embeddings[train_indices], labels[train_indices])
        predictions = model.predict(embeddings[test_indices])
        report = classification_report(labels[test_indices], predictions, output_dict=True, zero_division=0)
        classes = np.unique(labels)
        results[name] = {
            "model": model,
            "metrics": pd.DataFrame([
                {"Category": label, **values} for label, values in report.items() if isinstance(values, dict)
            ]),
            "confusion_matrix": pd.DataFrame(
                confusion_matrix(labels[test_indices], predictions, labels=classes),
                index=classes,
                columns=classes,
            ),
            "predictions": predictions,
            "test_indices": test_indices,
        }
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(labels[test_indices], predictions),
            "Macro F1": report["macro avg"]["f1-score"],
            "Weighted F1": report["weighted avg"]["f1-score"],
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False), results


def cosine_recommendations(
    embeddings: np.ndarray,
    dataframe: pd.DataFrame,
    source_index: int,
    count: int = 5,
) -> pd.DataFrame:
    """Return the most similar papers to one selected paper using cosine similarity."""
    if len(embeddings) != len(dataframe):
        raise ValueError("Embedding rows do not match the dataset rows.")
    if source_index < 0 or source_index >= len(dataframe):
        raise IndexError("The selected paper index is outside the dataset.")
    matrix = normalize(np.asarray(embeddings, dtype=np.float32))
    scores = matrix @ matrix[source_index]
    scores[source_index] = -np.inf
    result_indices = np.argsort(scores)[::-1][:count]
    result = dataframe.iloc[result_indices].copy()
    result.insert(0, "Similarity", scores[result_indices])
    result.insert(1, "Source index", result_indices)
    return result


def trend_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Count papers by year and category."""
    valid = dataframe.dropna(subset=["analysis_year"])
    if valid.empty:
        return pd.DataFrame(columns=["Year", "Category", "Papers"])
    return (
        valid.groupby(["analysis_year", "analysis_category"], as_index=False).size()
        .rename(columns={"analysis_year": "Year", "analysis_category": "Category", "size": "Papers"})
        .sort_values(["Year", "Papers"])
    )


def keyword_table(dataframe: pd.DataFrame, count: int = 25) -> pd.DataFrame:
    """Return keyword frequencies using the same token style as Word2Vec."""
    counts = Counter(token for text in dataframe["analysis_text"] for token in TOKEN_PATTERN.findall(text.lower()) if len(token) > 2)
    return pd.DataFrame(counts.most_common(count), columns=["Keyword", "Frequency"])


def potential_research_gaps(dataframe: pd.DataFrame, count: int = 20) -> pd.DataFrame:
    """Flag rare and recent word pairs as potential directions for human review."""
    valid = dataframe.dropna(subset=["analysis_year"])
    if valid.empty:
        return pd.DataFrame(columns=["Potential combination", "Papers", "Latest year", "Reason"])
    recent_cutoff = int(valid["analysis_year"].max()) - 2
    records = []
    for _, row in valid.iterrows():
        tokens = TOKEN_PATTERN.findall(row["analysis_text"].lower())
        records.extend((" ".join(pair), int(row["analysis_year"])) for pair in set(zip(tokens, tokens[1:])))
    counts = Counter(term for term, _ in records)
    latest = {term: max(year for candidate, year in records if candidate == term) for term in counts}
    candidates = [(term, frequency, latest[term]) for term, frequency in counts.items()
                  if frequency <= max(3, int(len(valid) * 0.01)) and latest[term] >= recent_cutoff]
    candidates.sort(key=lambda item: (item[2], -item[1]), reverse=True)
    return pd.DataFrame([
        {"Potential combination": term, "Papers": frequency, "Latest year": year,
         "Reason": "Rare and recent; requires human verification."}
        for term, frequency, year in candidates[:count]
    ])
