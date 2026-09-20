"""Reusable inference service for the paper intelligence interface."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed_data"
DATASET_PATH = DATA_DIR / "preprocessed_research_papers.csv"
WORD_DOCUMENT_VECTORS = DATA_DIR / "document_vectors_improved.npy"
WORD_VECTORS = DATA_DIR / "word_vectors_improved.dat"
VOCABULARY_PATH = DATA_DIR / "word2vec_vocabulary.csv"
BERT_EMBEDDINGS = DATA_DIR / "distilbert_embeddings.npz"
MODEL_NAME = "distilbert-base-uncased"
VECTOR_SIZE = 300
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


@dataclass
class ModelResult:
    name: str
    prediction: str
    probabilities: dict[str, float]
    validation_accuracy: float
    available: bool = True
    error: str | None = None


@dataclass
class PaperRecommendation:
    title: str
    category: str
    similarity: float
    link: str | None
    citation_count: str
    cited_status: str


def _load_labels() -> np.ndarray:
    with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as data_file:
        rows = csv.DictReader(data_file)
        labels = [row["Category"].strip() for row in rows]
    labels_array = np.asarray(labels)
    if not len(labels_array):
        raise ValueError("The processed dataset does not contain any labels")
    return labels_array


def _split_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    return train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)


def _train_classifier(features: np.ndarray, labels: np.ndarray, indices: tuple[np.ndarray, np.ndarray]):
    train_indices, test_indices = indices
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_indices])
    test_features = scaler.transform(features[test_indices])
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    classifier.fit(train_features, labels[train_indices])
    accuracy = float(classifier.score(test_features, labels[test_indices]))
    return scaler, classifier, accuracy


def _probability_map(classifier: LogisticRegression, probabilities: np.ndarray) -> dict[str, float]:
    return {
        str(category): float(probability)
        for category, probability in sorted(
            zip(classifier.classes_, probabilities), key=lambda item: item[1], reverse=True
        )
    }


class PaperClassifier:
    """Load saved embeddings and classify paper title/abstract text."""

    def __init__(self) -> None:
        self.labels = _load_labels()
        self.classes = np.unique(self.labels)
        self.split_indices = _split_indices(self.labels)
        self.word_scaler = None
        self.word_classifier = None
        self.word_accuracy = 0.0
        self.bert_scaler = None
        self.bert_classifier = None
        self.bert_accuracy = 0.0
        self.tokenizer = None
        self.encoder = None
        self.document_vectors = np.asarray(
            np.load(WORD_DOCUMENT_VECTORS, mmap_mode="r"), dtype=np.float32
        )
        self.paper_rows = self._load_paper_rows()
        self._load_word2vec_classifier()
        self._load_bert_classifier()

    def _load_paper_rows(self) -> list[dict[str, str]]:
        with DATASET_PATH.open("r", encoding="utf-8-sig", newline="") as data_file:
            rows = list(csv.DictReader(data_file))
        if len(rows) != len(self.labels):
            raise ValueError("Dataset rows and document vectors have different row counts")
        return rows

    def _load_word2vec_classifier(self) -> None:
        vectors = np.asarray(np.load(WORD_DOCUMENT_VECTORS, mmap_mode="r"), dtype=np.float32)
        if vectors.shape[0] != len(self.labels):
            raise ValueError("Word2Vec document vectors and labels have different row counts")
        self.word_scaler, self.word_classifier, self.word_accuracy = _train_classifier(
            vectors, self.labels, self.split_indices
        )
        vocabulary_table: dict[str, int] = {}
        with VOCABULARY_PATH.open("r", encoding="utf-8-sig", newline="") as vocabulary_file:
            for row in csv.DictReader(vocabulary_file):
                vocabulary_table[row["word"]] = int(row["word_id"])
        self.word_to_id = vocabulary_table
        vector_rows = max(vocabulary_table.values()) + 1
        self.word_vectors = np.memmap(
            WORD_VECTORS, mode="r", dtype="float32", shape=(vector_rows, VECTOR_SIZE)
        )

    def _load_bert_classifier(self) -> None:
        cached = np.load(BERT_EMBEDDINGS, allow_pickle=False)
        embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
        bert_labels = self.labels
        if embeddings.shape[0] != len(self.labels):
            random_generator = np.random.default_rng(42)
            selected_indices: list[int] = []
            rows_per_category = max(1, embeddings.shape[0] // len(self.classes))
            for category in self.classes:
                category_indices = np.flatnonzero(self.labels == category)
                random_generator.shuffle(category_indices)
                selected_indices.extend(category_indices[:rows_per_category].tolist())
            random_generator.shuffle(selected_indices)
            selected_indices = selected_indices[: embeddings.shape[0]]
            if len(selected_indices) != embeddings.shape[0]:
                raise ValueError("DistilBERT cache cannot be aligned with dataset labels")
            bert_labels = self.labels[selected_indices]
        bert_split_indices = _split_indices(bert_labels)
        self.bert_scaler, self.bert_classifier, self.bert_accuracy = _train_classifier(
            embeddings, bert_labels, bert_split_indices
        )

    def _word_vector(self, text: str) -> np.ndarray:
        token_vectors = [
            self.word_vectors[self.word_to_id[token]]
            for token in TOKEN_PATTERN.findall(text.lower())
            if token in self.word_to_id
        ]
        if not token_vectors:
            raise ValueError("No words from the input were found in the Word2Vec vocabulary")
        return np.mean(token_vectors, axis=0, dtype=np.float32).reshape(1, -1)

    @staticmethod
    def _paper_link(row: dict[str, str]) -> str | None:
        for field in ("Link", "link", "URL", "url", "DOI", "doi"):
            value = (row.get(field) or "").strip()
            if value:
                if field.lower() == "doi" and not value.lower().startswith("http"):
                    return f"https://doi.org/{value.removeprefix('doi:').strip()}"
                return value if value.lower().startswith("http") else f"https://{value}"
        return None

    def recommend(
        self, title: str, abstract: str, category: str, limit: int = 5
    ) -> list[PaperRecommendation]:
        query_text = abstract.strip() or title.strip()
        if not query_text:
            return []
        category_indices = np.flatnonzero(self.labels == category)
        candidate_indices = []
        candidate_texts = []
        for index in category_indices:
            paper_abstract = (self.paper_rows[int(index)].get("Abstract") or "").strip()
            if not paper_abstract:
                continue
            candidate_indices.append(index)
            candidate_texts.append(paper_abstract)
        if not candidate_texts:
            return []
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        vectors = vectorizer.fit_transform([query_text, *candidate_texts])
        query_vector = vectors[0]
        candidate_vectors = vectors[1:]
        similarities = (candidate_vectors @ query_vector.T).toarray().ravel()
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
        ranked_positions = np.argsort(similarities)[::-1][:limit]
        recommendations = []
        for position in ranked_positions:
            row = self.paper_rows[int(category_indices[position])]
            recommendations.append(
                PaperRecommendation(
                    title=(row.get("Title") or "Untitled paper").strip(),
                    category=(row.get("Category") or category).strip(),
                    similarity=float(similarities[position]),
                    link=self._paper_link(row),
                    citation_count=(row.get("Citation count") or "").strip(),
                    cited_status=(
                        "Top 1% cited"
                        if (row.get("Top 1% cited") or "").strip().lower() in {"1", "yes", "true", "y"}
                        else "Top 10% cited"
                        if (row.get("Top 10% cited") or "").strip().lower() in {"1", "yes", "true", "y"}
                        else ""
                    ),
                )
            )
        return recommendations

    def _bert_vector(self, text: str) -> np.ndarray:
        if self.tokenizer is None or self.encoder is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.encoder = AutoModel.from_pretrained(MODEL_NAME)
            self.encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.encoder.eval()

        import torch

        device = next(self.encoder.parameters()).device
        with torch.no_grad():
            batch = self.tokenizer(
                [text], padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(device)
            output = self.encoder(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            vector = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return vector.cpu().numpy().astype(np.float32)

    def predict(self, text: str) -> list[ModelResult]:
        results: list[ModelResult] = []
        word_features = self.word_scaler.transform(self._word_vector(text))
        word_probabilities = self.word_classifier.predict_proba(word_features)[0]
        results.append(
            ModelResult(
                "Word2Vec",
                str(self.word_classifier.classes_[np.argmax(word_probabilities)]),
                _probability_map(self.word_classifier, word_probabilities),
                self.word_accuracy,
            )
        )
        try:
            bert_features = self.bert_scaler.transform(self._bert_vector(text))
            bert_probabilities = self.bert_classifier.predict_proba(bert_features)[0]
            results.append(
                ModelResult(
                    "DistilBERT",
                    str(self.bert_classifier.classes_[np.argmax(bert_probabilities)]),
                    _probability_map(self.bert_classifier, bert_probabilities),
                    self.bert_accuracy,
                )
            )
        except Exception as error:
            results.append(
                ModelResult("DistilBERT", "Unavailable", {}, self.bert_accuracy, False, str(error))
            )
        return results