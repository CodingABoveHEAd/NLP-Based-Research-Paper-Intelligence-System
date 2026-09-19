"""Streamlit interface for classification, trends, and potential gap analysis."""

from pathlib import Path
import sys

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from prince_analysis import (  # noqa: E402
    cosine_recommendations,
    evaluate_embedding_models,
    keyword_table,
    load_embedding,
    normalize_dataset,
    potential_research_gaps,
    trend_table,
)


st.set_page_config(page_title="Paper Intelligence", page_icon="P", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#17202a; --muted:#68737d; --cream:#f7f3ec; --mint:#dcefe8; --coral:#e87561; --navy:#153d4d; }
    .stApp { background:var(--cream); color:var(--ink); font-family:'DM Sans', sans-serif; }
    h1,h2,h3 { font-family:'Space Grotesk', sans-serif; color:var(--navy); letter-spacing:0; }
    h1 { font-size:2.7rem; margin-bottom:.2rem; }
    .hero { padding:2.2rem 2.4rem; border-radius:18px; background:linear-gradient(120deg,#dcefe8 0%,#f7f3ec 58%,#f4d2c9 100%); border:1px solid #cbded6; margin-bottom:1.4rem; }
    .hero p { color:#52616b; font-size:1.05rem; max-width:760px; }
    .metric { background:#fffdf9; border:1px solid #e1ddd4; border-radius:12px; padding:1rem; }
    .stButton>button { background:var(--navy); color:white; border:0; border-radius:9px; }
    [data-testid='stFileUploader'] { background:#fffdf9; border-radius:12px; padding:.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div class='hero'><h1>Paper Intelligence</h1><p>A focused workspace for classifying research papers, reading publication patterns, and surfacing potential research directions.</p></div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a research-paper CSV", type="csv")
    default_path = Path(__file__).parent / "data" / "processed_data" / "preprocessed_research_papers.csv"
    st.caption("Expected labels: Category, category, or Topic. Text can be text, clean_text, or Title + Abstract.")

try:
    if uploaded is not None:
        papers = normalize_dataset(pd.read_csv(uploaded))
    elif default_path.exists():
        papers = normalize_dataset(pd.read_csv(default_path))
    else:
        st.info("Upload the processed CSV from the sidebar to begin.")
        st.stop()
except Exception as error:
    st.error(f"Could not load this dataset: {error}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Papers", f"{len(papers):,}")
col2.metric("Categories", papers["analysis_category"].nunique())
col3.metric("With year", int(papers["analysis_year"].notna().sum()))

tab_classify, tab_recommend, tab_trends, tab_gaps = st.tabs(["Classify", "Recommend", "Trends & keywords", "Potential gaps"])
with tab_classify:
    st.subheader("Word2Vec and BERT Logistic Regression")
    artifact_paths = {
        "Word2Vec": [
            Path(__file__).parent / "data" / "processed_data" / "document_vectors_improved.npy",
            Path(__file__).parent / "data" / "processed_data" / "document_vectors.npy",
        ],
        "BERT": [
            Path(__file__).parent / "data" / "processed_data" / "distilbert_embeddings.npy",
            Path(__file__).parent / "data" / "processed_data" / "distilbert_embeddings.npz",
        ],
    }
    available = {name: next((path for path in paths if path.exists()), None) for name, paths in artifact_paths.items()}

    def read_artifact(path, row_count):
        if path.suffix == ".npz":
            values = np.load(path, allow_pickle=False)["embeddings"]
            if values.shape[0] != row_count:
                raise ValueError("The saved BERT embeddings do not match the dataset row count.")
            return values
        return load_embedding(path, row_count)

    loaded_embeddings = {}
    for name, path in available.items():
        if path is not None:
            try:
                loaded_embeddings[name] = read_artifact(path, len(papers))
            except (ValueError, KeyError) as error:
                st.warning(f"{name} embedding unavailable: {error}")

    if not loaded_embeddings:
        st.info("Run the Word2Vec and BERT notebooks/scripts first. Their saved embedding files will appear here automatically.")
    else:
        try:
            comparison, evaluations = evaluate_embedding_models(loaded_embeddings, papers["analysis_category"])
            st.write("Both models use the same stratified 80/20 split.")
            st.dataframe(comparison.style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}", "Weighted F1": "{:.3f}"}), use_container_width=True, hide_index=True)
            selected = st.selectbox("Inspect model", list(evaluations))
            selected_evaluation = evaluations[selected]
            st.dataframe(selected_evaluation["metrics"], use_container_width=True, hide_index=True)
            st.write("Confusion matrix")
            st.dataframe(selected_evaluation["confusion_matrix"], use_container_width=True)
            paper_index = st.number_input("Paper index to classify", min_value=0, max_value=len(papers) - 1, value=0, step=1)
            paper_vector = loaded_embeddings[selected][int(paper_index)].reshape(1, -1)
            probabilities = selected_evaluation["model"].predict_proba(paper_vector)[0]
            prediction = selected_evaluation["model"].classes_[int(np.argmax(probabilities))]
            st.success(f"Prediction: {prediction}")
            st.dataframe(pd.DataFrame({"Category": selected_evaluation["model"].classes_, "Probability": probabilities}).style.format({"Probability": "{:.1%}"}), use_container_width=True, hide_index=True)
        except ValueError as error:
            st.warning(str(error))

with tab_recommend:
    st.subheader("Similar research papers")
    available_recommendation_models = list(loaded_embeddings)
    if not available_recommendation_models:
        st.info("Run the Word2Vec or BERT embedding pipeline first to enable recommendations.")
    else:
        recommendation_model = st.selectbox("Embedding model for similarity", available_recommendation_models)
        source_index = st.number_input("Source paper index", min_value=0, max_value=len(papers) - 1, value=0, step=1)
        count = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
        source = papers.iloc[int(source_index)]
        st.write(f"**Source:** {source.get('Title', source.get('analysis_text', 'Selected paper'))}")
        recommendations = cosine_recommendations(loaded_embeddings[recommendation_model], papers, int(source_index), count)
        display_columns = [column for column in ["Similarity", "Title", "analysis_category", "Abstract", "DOI"] if column in recommendations]
        st.dataframe(recommendations[display_columns].style.format({"Similarity": "{:.3f}"}), use_container_width=True, hide_index=True)

with tab_trends:
    st.subheader("Publication patterns")
    terms = keyword_table(papers)
    st.dataframe(terms, use_container_width=True, hide_index=True)
    trend = trend_table(papers)
    if trend.empty:
        st.info("No usable publication year column was found.")
    else:
        chart = px.line(trend, x="Year", y="Papers", color="Category", markers=True, template="simple_white")
        chart.update_layout(legend_title_text="Category", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(trend, use_container_width=True, hide_index=True)

with tab_gaps:
    st.subheader("Potential research directions")
    st.caption("These are signals for human review, not confirmed research gaps.")
    gaps = potential_research_gaps(papers)
    if gaps.empty:
        st.info("Gap analysis needs a usable year column and text containing repeated word pairs.")
    else:
        st.dataframe(gaps, use_container_width=True, hide_index=True)