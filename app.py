from __future__ import annotations

import re

import streamlit as st
from pypdf import PdfReader

from src.paper_classifier import PaperClassifier


st.set_page_config(page_title="Paper Intelligence", page_icon="P", layout="wide")


@st.cache_resource(show_spinner=False)
def load_classifier() -> PaperClassifier:
    return PaperClassifier()


def extract_pdf_title_abstract(uploaded_file) -> tuple[str, str]:
    reader = PdfReader(uploaded_file)
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("No selectable text was found in this PDF")
    abstract_match = re.search(
        r"\babstract\b\s*[:.-]?\s*(.*?)(?=\n\s*(?:keywords?|introduction|1\.?\s+introduction)\b|$)",
        text,
        re.I | re.S,
    )
    abstract = re.sub(r"\s+", " ", abstract_match.group(1)).strip() if abstract_match else ""
    metadata_title = str((reader.metadata or {}).get("/Title") or "").strip()
    metadata_title = re.sub(r"\s+", " ", metadata_title)
    before_abstract = text[: abstract_match.start()] if abstract_match else text[:3000]
    lines = [re.sub(r"\s+", " ", line).strip() for line in before_abstract.splitlines()]
    lines = [line for line in lines if line]
    ignored_patterns = re.compile(
        r"^(abstract|keywords?|introduction|arxiv:|doi:|www\.|https?://)|"
        r"\b(university|institute|department|college|@)\b|\b\d{4}\b",
        re.I,
    )
    title_candidates = [
        line for line in lines
        if 20 <= len(line) <= 240 and not ignored_patterns.search(line)
    ]
    if metadata_title and not re.match(r"^(microsoft word|untitled|document)$", metadata_title, re.I):
        title = metadata_title
    elif title_candidates:
        title = max(title_candidates[:8], key=lambda line: (len(line), -lines.index(line)))
    else:
        title = lines[0] if lines else ""
    return title, abstract


def render_result(result) -> None:
    st.subheader(f"{result.name} result")
    if not result.available:
        st.warning(f"{result.name} could not process this text: {result.error}")
        return
    st.metric("Predicted category", result.prediction)
    st.caption(f"Held-out validation accuracy: {result.validation_accuracy:.1%}")
    for category, probability in result.probabilities.items():
        st.write(f"**{category}** · {probability:.1%}")
        st.progress(probability)


def render_recommendations(recommendations) -> None:
    st.subheader("Related papers from the dataset")
    st.caption("Ranked by cosine similarity within the predicted category.")
    for index, paper in enumerate(recommendations, start=1):
        st.markdown(f"**{index}. {paper.title}**")
        st.write(f"Similarity: {paper.similarity:.1%} · Category: {paper.category}")
        if paper.link:
            st.markdown(f"[Open paper]({paper.link})")
        else:
            st.caption("No paper link is available in the dataset.")


st.title("Research Paper Intelligence")
st.write("Upload a paper or enter a title and abstract to identify its research category.")
input_mode = st.radio("Input source", ["Write title and abstract", "Upload PDF"], horizontal=True)
title = ""
abstract = ""

if input_mode == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file is not None and st.button("Extract text", type="secondary"):
        try:
            title, abstract = extract_pdf_title_abstract(uploaded_file)
            st.session_state["title"] = title
            st.session_state["abstract"] = abstract
            st.success("Title and abstract extracted. Review them below before classifying.")
        except Exception as error:
            st.error(f"Could not read this PDF: {error}")
    title = st.text_input("Title", value=st.session_state.get("title", ""))
    abstract = st.text_area("Abstract (optional)", value=st.session_state.get("abstract", ""), height=220)
else:
    title = st.text_input("Title")
    abstract = st.text_area("Abstract (optional)", height=220)

recommendation_count = st.selectbox("Number of related papers", options=[3, 5], index=1)

if st.button("Classify paper", type="primary", use_container_width=True):
    query = f"{title.strip()} {abstract.strip()}".strip()
    if not query:
        st.error("Enter a title or abstract first.")
    else:
        try:
            with st.spinner("Comparing Word2Vec and DistilBERT classifiers..."):
                classifier = load_classifier()
                results = classifier.predict(query)
            available_results = [result for result in results if result.available]
            best_result = max(available_results, key=lambda result: result.validation_accuracy)
            st.success(f"Recommended category: {best_result.prediction}")
            st.caption("The recommended model is selected by held-out dataset accuracy.")
            columns = st.columns(len(results))
            for column, result in zip(columns, results):
                with column:
                    render_result(result)
            recommendations = classifier.recommend(
                title, abstract, best_result.prediction, limit=recommendation_count
            )
            if recommendations:
                render_recommendations(recommendations)
            else:
                st.info("No related papers were found for this category.")
        except Exception as error:
            st.error(f"Classification failed: {error}")