"""BERT embeddings with a built-in multiclass logistic-regression classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed_data" / "preprocessed_research_papers.csv"
CACHE_PATH = PROJECT_ROOT / "data" / "processed_data" / "distilbert_embeddings.npz"
MODEL_NAME = "distilbert-base-uncased"


def load_dataset(path: Path, max_rows: int | None = None) -> tuple[list[str], np.ndarray]:
    """Load text and Category labels without reading unrelated CSV columns."""
    texts: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as data_file:
        reader = csv.DictReader(data_file)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("Category") or "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    if max_rows is not None and len(texts) > max_rows:
        random_generator = np.random.default_rng(42)
        all_labels = np.asarray(labels)
        selected_indices: list[int] = []
        categories = np.unique(all_labels)
        rows_per_category = max(1, max_rows // len(categories))
        for category in categories:
            category_indices = np.flatnonzero(all_labels == category)
            random_generator.shuffle(category_indices)
            selected_indices.extend(category_indices[:rows_per_category].tolist())
        random_generator.shuffle(selected_indices)
        selected_indices = selected_indices[:max_rows]
        texts = [texts[index] for index in selected_indices]
        labels = [labels[index] for index in selected_indices]
    if len(set(labels)) < 2:
        raise ValueError("The dataset must contain at least two non-empty categories")
    return texts, np.asarray(labels)


def mean_pool(last_hidden_state, attention_mask):
    """Average only non-padding token vectors."""
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def encode_texts(
    texts: list[str],
    tokenizer,
    encoder,
    batch_size: int = 16,
    max_length: int = 256,
) -> np.ndarray:
    """Create contextual DistilBERT document vectors in batches."""
    import torch

    device = next(encoder.parameters()).device
    vectors: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = encoder(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors).astype(np.float32)


def load_or_create_embeddings(
    texts: list[str],
    model_name: str = MODEL_NAME,
    cache_path: Path = CACHE_PATH,
    batch_size: int = 16,
    max_length: int = 256,
) -> np.ndarray:
    """Load cached BERT vectors or download the encoder and create them."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if cache_path.exists():
        print(f"Loading cached embeddings: {cache_path}", flush=True)
        cached = np.load(cache_path, allow_pickle=False)
        if cached["count"].item() == len(texts) and cached["model"].item() == model_name:
            return cached["embeddings"]

    print(f"Loading pretrained model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    print(f"Encoding {len(texts)} documents with batch size {batch_size}", flush=True)
    embeddings = encode_texts(texts, tokenizer, encoder, batch_size, max_length)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        count=np.asarray(len(texts)),
        model=np.asarray(model_name),
    )
    return embeddings


def train_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train built-in balanced multiclass logistic regression."""
    train_x, test_x, train_y, test_y = train_test_split(
        embeddings,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    classifier = LogisticRegression(
        max_iter=1_000,
        class_weight="balanced",
        solver="lbfgs",
    )
    classifier.fit(train_x, train_y)
    return classifier, train_x, train_y, test_x, test_y


def print_evaluation(classifier: LogisticRegression, test_x: np.ndarray, test_y: np.ndarray) -> None:
    predictions = classifier.predict(test_x)
    print(f"Accuracy: {accuracy_score(test_y, predictions):.4f}")
    print(classification_report(test_y, predictions, zero_division=0))


def predict_text(
    text: str,
    classifier: LogisticRegression,
    tokenizer,
    encoder,
) -> None:
    vector = encode_texts([text], tokenizer, encoder)
    prediction = classifier.predict(vector)[0]
    probabilities = classifier.predict_proba(vector)[0]
    print(f"Text: {text}")
    print(f"Predicted category: {prediction}")
    print("Probabilities:")
    for category, probability in sorted(
        zip(classifier.classes_, probabilities), key=lambda item: item[1], reverse=True
    ):
        print(f"  {category}: {probability:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT embeddings plus built-in logistic regression.")
    parser.add_argument("--text", help="Classify one title or abstract after training.")
    parser.add_argument("--max-rows", type=int, default=2_000, help="Rows used for the demo run.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    args = parser.parse_args()

    print(f"Loading dataset: {DATASET_PATH}", flush=True)
    texts, labels = load_dataset(DATASET_PATH, args.max_rows)
    print(f"Loaded {len(labels)} labeled documents", flush=True)
    embeddings = load_or_create_embeddings(
        texts,
        cache_path=args.cache,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print("Training built-in logistic regression", flush=True)
    classifier, _, _, test_x, test_y = train_classifier(embeddings, labels)
    print(f"Model: {MODEL_NAME}")
    print(f"Rows: {len(labels)}")
    print(f"Embedding shape: {embeddings.shape}")
    print_evaluation(classifier, test_x, test_y)

    if args.text:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        encoder = AutoModel.from_pretrained(MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.to(device)
        predict_text(args.text, classifier, tokenizer, encoder)


if __name__ == "__main__":
    main()