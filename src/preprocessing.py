"""Simple preprocessing helpers for research-paper title and abstract text."""

import re
from typing import Iterable

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

REQUIRED_COLUMNS = [
    "Title",
    "Author",
    "Institution",
    "Topic",
    "Domain",
    "Field",
    "Concept",
    "Abstract",
    "Keyword",
    "Citation count",
    "Cites",
    "Top 1% cited",
    "Top 10% cited",
    "DOI",
]

OUTPUT_COLUMNS = REQUIRED_COLUMNS.copy()
PREPROCESS_COLUMNS = [
    "Title",
    "Topic",
    "Domain",
    "Field",
    "Concept",
    "Abstract",
    "Keyword",
    "Category",
]

# Keep domain terms that may also be treated as ordinary short words.
TECHNICAL_TERMS = {"bert", "cnn", "hpc", "gpu", "nlp", "llm"}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
LEMMATIZER = WordNetLemmatizer()

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
        "was", "were", "with",
    }

STOP_WORDS -= TECHNICAL_TERMS


def citation_flag(value: object) -> int:
    """Convert common yes/no citation labels to the required 1/0 format."""
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    label = str(value).strip().lower()
    if label.startswith("not ") or label in {"no", "n", "false", "0"}:
        return 0
    return int(label in {"yes", "y", "true", "1"} or label.startswith("top "))


def tokenize_text(text: str) -> list[str]:
    """Lowercase text and split it into words, numbers, and technical terms."""
    if not isinstance(text, str):
        return []
    return TOKEN_PATTERN.findall(text.lower())


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Remove English stopwords while retaining important technical terms."""
    return [token for token in tokens if token not in STOP_WORDS]


def clean_text(text: str, lemmatize: bool = False) -> str:
    """Apply lowercase, tokenization, punctuation filtering, and stopword removal."""
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    if lemmatize:
        try:
            tokens = [LEMMATIZER.lemmatize(token) for token in tokens]
        except LookupError:
            pass
    return " ".join(tokens)


def preprocess_document(title: object, abstract: object, lemmatize: bool = False) -> dict[str, str]:
    """Create the clean text fields for one paper from its title and abstract."""
    title = "" if pd.isna(title) else str(title).strip()
    abstract = "" if pd.isna(abstract) else str(abstract).strip()
    return {
        "title": title,
        "abstract": abstract,
        "clean_text": clean_text(title + " " + abstract, lemmatize=lemmatize),
    }


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Raise an error when the input does not contain the project schema."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def preprocess_dataframe(dataframe: pd.DataFrame, lemmatize: bool = False) -> pd.DataFrame:
    """Keep all original paper columns and add clean text output columns."""
    validate_required_columns(dataframe)
    processed = dataframe.copy()
    documents = processed.apply(
        lambda row: preprocess_document(row["Title"], row["Abstract"], lemmatize=lemmatize),
        axis=1,
        result_type="expand",
    )
    processed[["title", "abstract", "clean_text"]] = documents[["title", "abstract", "clean_text"]]
    processed["category"] = processed["Topic"].fillna("").astype(str).str.strip()
    year_column = next(
        (column for column in ("year", "Year", "Publication year", "Publication Year", "publication_year") if column in processed),
        None,
    )
    processed["year"] = (
        pd.to_numeric(processed[year_column], errors="coerce").astype("Int64")
        if year_column
        else pd.Series(pd.NA, index=processed.index, dtype="Int64")
    )
    return processed


def build_preprocessed_dataset(dataframe: pd.DataFrame, lemmatize: bool = False) -> pd.DataFrame:
    """Create the export while preserving source columns and adding combined text."""
    validate_required_columns(dataframe)
    output = dataframe.copy()

    for column in PREPROCESS_COLUMNS:
        if column in output.columns:
            output[column] = output[column].fillna("").map(
                lambda value: clean_text(str(value), lemmatize=lemmatize)
            )

    output["text"] = (
        output["Title"].fillna("").astype(str).str.strip()
        + " "
        + output["Abstract"].fillna("").astype(str).str.strip()
    ).str.strip()

    output["Top 1% cited"] = output["Top 1% cited"].map(citation_flag)
    output["Top 10% cited"] = output["Top 10% cited"].map(citation_flag)
    return output
