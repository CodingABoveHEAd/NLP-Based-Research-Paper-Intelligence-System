# NLP-Based Research Paper Intelligence System
## Project Report

## Abstract

The NLP-Based Research Paper Intelligence System is an end-to-end natural language processing application for classifying research papers and helping users discover related literature. The system accepts either a research-paper PDF or manually entered title and abstract text. It extracts the available paper content, predicts the most likely research category, displays the probability distribution across categories, and recommends related papers from the project dataset.

Two document-representation approaches are included in the project. The first uses averaged Word2Vec representations and a balanced multiclass logistic-regression classifier. The second uses cached DistilBERT embeddings with a logistic-regression classifier. The application evaluates both available model paths using held-out validation data and identifies the stronger available classifier for the final category recommendation.

After classification, the system retrieves candidate papers from the predicted category and ranks them using TF-IDF cosine similarity between the user's abstract and the candidate papers' abstracts. Each recommendation includes its similarity score, citation count, optional top-cited status, and an available DOI or paper URL.

---

## 1. Introduction

The amount of research literature available to students, researchers, and professionals is continuously increasing. Finding the right papers requires more than a keyword search: a user may need to understand the research area of a paper, compare it with existing work, and identify the most relevant related studies.

This project addresses that need through a practical research-paper intelligence workflow. It combines classical word embeddings, contextual transformer embeddings, supervised classification, and similarity-based retrieval in a single interface. The result is a system that can be used both as a machine-learning experiment and as a usable research-support application.

The main purpose of the project is not to replace expert literature review. Instead, it provides a fast first-pass assistant that can organize an input paper, suggest its likely category, and surface relevant documents from the available dataset.

## 1.1 Problem Statement

Given a research paper represented by its title and abstract, the system should:

1. Extract title and abstract text from a PDF when a PDF is provided.
2. Accept title and abstract text directly from a user.
3. Predict the paper's research category.
4. Compare Word2Vec and DistilBERT-based classifiers.
5. Show category probabilities for the available models.
6. Retrieve and rank related papers from the project dataset.
7. Display useful metadata, including citation count, top-cited status, and paper links.

## 1.2 Project Objectives

The project was designed with the following objectives:

- Build a complete preprocessing workflow for research-paper data.
- Create document representations using Word2Vec and DistilBERT.
- Implement multiclass logistic regression for category prediction.
- Provide a usable interface rather than limiting the work to notebooks.
- Rank related documents based on abstract-level textual similarity.
- Preserve useful research metadata in the final user experience.

---

## 2. System Overview

The system is organized into five logical stages:

```text
Raw research-paper data
        |
        v
Dataset merging and preprocessing
        |
        v
Word2Vec and DistilBERT document representations
        |
        v
Balanced multiclass logistic-regression classifiers
        |
        v
Streamlit classification and recommendation interface
```

The user-facing workflow is:

```text
PDF upload or manual text
        |
        v
Title and abstract extraction/input
        |
        v
Classifier inference
        |
        v
Category probabilities and recommended category
        |
        v
Category-filtered abstract similarity search
        |
        v
Top-3 or Top-5 related papers with metadata and links
```

The application is implemented as a local Streamlit application. The model service is separated from the interface so that the classification and recommendation logic can be reused independently of the web presentation layer.

---

## 3. Repository Structure

```text
NLP-Based-Research-Paper-Intelligence-System/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- PROJECT_REPORT.md
|
|-- data/
|   |-- raw_data/
|   |   |-- hpc.csv
|   |   |-- iot.csv
|   |   |-- networks.csv
|   |   |-- nlp.csv
|   |   |-- security.csv
|   |   |-- vision.csv
|   |
|   |-- processed_data/
|       |-- merged_research_papers.csv
|       |-- preprocessed_research_papers.csv
|       |-- document_vectors.npy
|       |-- document_vectors_improved.npy
|       |-- word2vec_vocabulary.csv
|       |-- word_vectors.dat
|       |-- word_vectors_improved.dat
|       |-- distilbert_embeddings.npz
|       |-- distilbert_demo_embeddings.npz
|
|-- notebooks/
|   |-- word2vec.ipynb
|   |-- logistic_regression.ipynb
|
|-- src/
|   |-- preprocessing.py
|   |-- paper_classifier.py
|   |-- bert_classifier.py
|
|-- utilitites/
    |-- merge_csv.ipynb
```

### 3.1 Main Application File

`app.py` is the Streamlit entry point. It controls the user interface, accepts input, performs PDF extraction, calls the classifier service, and displays the classification and recommendation results.

### 3.2 Reusable Classifier Service

`src/paper_classifier.py` contains the application's central inference service. It loads labels and saved representations, trains the logistic-regression models, creates vectors for new input, calculates probabilities, and produces related-paper recommendations.

### 3.3 Preprocessing Module

`src/preprocessing.py` contains reusable text-cleaning and dataset-validation functions. It is responsible for tokenization, stop-word handling, optional lemmatization, citation flag normalization, and construction of combined title-and-abstract text.

### 3.4 Standalone BERT Classifier

`src/bert_classifier.py` provides a command-line DistilBERT embedding and logistic-regression workflow. It can load or create cached embeddings, train a classifier, print evaluation metrics, and classify a custom text from the command line.

### 3.5 Notebooks

The notebooks document the experimental stages of the project. The Word2Vec notebook covers word and document representation. The logistic-regression notebook demonstrates the manual multiclass softmax implementation, evaluation, custom prediction, and BERT comparison workflow.

---

## 4. Dataset and Data Preparation

## 4.1 Raw Dataset

The `data/raw_data/` directory contains separate CSV files for the project's research areas:

- High-performance computing (`hpc.csv`)
- Internet of Things (`iot.csv`)
- Computer networks (`networks.csv`)
- Natural language processing (`nlp.csv`)
- Cybersecurity (`security.csv`)
- Computer vision (`vision.csv`)

The separate files are merged into a common research-paper collection before the machine-learning stages are applied.

## 4.2 Processed Dataset

The main application dataset is `data/processed_data/preprocessed_research_papers.csv`. It preserves paper metadata and contains the fields needed by the classifier and recommendation system.

Important fields include:

- `Title`: the paper title.
- `Abstract`: the paper abstract.
- `Category`: the classification label used by the models.
- `Citation count`: the number of citations shown with recommendations.
- `Top 1% cited`: an indicator for highly cited papers.
- `Top 10% cited`: an indicator for papers in the top ten percent by citations.
- `DOI`: the paper identifier used to construct a direct DOI link.
- `text`: the combined title and abstract field used by the embedding workflow.

The application expects the row order of the processed CSV to remain aligned with the saved document-vector arrays. This alignment is important because each vector must correspond to the correct paper metadata and category label.

## 4.3 Text Preprocessing

The preprocessing module applies the following operations:

1. Convert text to a consistent lowercase form.
2. Tokenize words, numbers, and technical terms using a regular expression.
3. Remove English stop words.
4. Preserve selected technical terms, including `bert`, `cnn`, `hpc`, `gpu`, `nlp`, and `llm`.
5. Optionally apply WordNet lemmatization.
6. Combine title and abstract text into a single document text.
7. Validate that required research-paper columns exist.
8. Normalize citation flags into binary values when preprocessing citation fields.

This design attempts to remove common linguistic noise while retaining terms that are important for technical research classification.

---

## 5. Word2Vec Representation and Classifier

## 5.1 Word-Level Representation

Word2Vec represents each vocabulary term as a dense numerical vector. Semantically related words tend to have similar vector positions because the model learns from contextual co-occurrence patterns.

The project stores the Word2Vec vocabulary in `word2vec_vocabulary.csv`. The word-vector binary files contain the corresponding numerical representations. For a new paper, the system identifies known tokens and averages their word vectors to create a fixed-size document vector.

Mathematically, if a document contains known word vectors $v_1, v_2, ..., v_n$, the document representation is:

$$
 d = \frac{1}{n} \sum_{i=1}^{n} v_i
$$

This produces one vector for a document regardless of the number of words it contains.

## 5.2 Saved Document Vectors

The project contains precomputed document representations in:

- `document_vectors.npy`
- `document_vectors_improved.npy`

The application uses the improved document-vector artifact. These vectors are loaded using NumPy memory mapping where appropriate, which reduces unnecessary memory duplication during application startup.

## 5.3 Logistic Regression

The Word2Vec document vectors are used as input features for a multiclass logistic-regression classifier. Before training, the features are standardized using `StandardScaler`.

The classifier uses:

- Multiclass logistic regression.
- The `lbfgs` solver.
- Balanced class weights.
- A fixed random seed of `42`.
- A maximum of 1,000 iterations.

Balanced class weights are important because research categories may not contain the same number of papers. Without balancing, a model can become biased toward the largest category.

## 5.4 Word2Vec Input Prediction

For a user query, the service:

1. Tokenizes the input text.
2. Keeps tokens present in the Word2Vec vocabulary.
3. Averages their vectors.
4. Applies the training scaler.
5. Generates class probabilities using logistic regression.
6. Selects the category with the largest probability.

If no input words are present in the vocabulary, the Word2Vec path raises a clear error rather than silently producing an invalid vector.

---

## 6. DistilBERT Representation and Classifier

## 6.1 Contextual Embeddings

DistilBERT is a smaller transformer-based language model derived from BERT. Unlike Word2Vec, which assigns one fixed vector to each word, DistilBERT generates contextual token representations. The meaning of a word can therefore vary according to the words around it.

The project uses `distilbert-base-uncased` through the Transformers library. The tokenizer converts text into model input tokens, and the encoder produces contextual hidden states.

## 6.2 Mean Pooling

The token-level hidden states are converted into one document vector using attention-mask-aware mean pooling. Padding tokens are excluded from the average. If the hidden states are $h_1, h_2, ..., h_m$ and the attention mask values are $a_i$, the pooled representation is:

$$
 d = \frac{\sum_{i=1}^{m} a_i h_i}{\sum_{i=1}^{m} a_i}
$$

This produces a fixed-size representation for the entire paper text.

## 6.3 Cached Embeddings

The file `distilbert_embeddings.npz` contains cached document embeddings. Using cached embeddings avoids re-encoding the entire dataset every time the classifier is initialized. The application still loads the pretrained DistilBERT model when it needs to encode a new user query.

The standalone BERT classifier can also create and save embeddings when a compatible cache is not available. The cache stores the embedding matrix, row count, and model name for validation.

## 6.4 DistilBERT Classification

The cached embeddings are used with a balanced multiclass logistic-regression classifier. The classification process follows the same general structure as the Word2Vec path:

1. Load contextual document embeddings.
2. Align their labels with the available embedding rows.
3. Split the data into training and testing subsets.
4. Standardize the features.
5. Train balanced logistic regression.
6. Generate class probabilities for new input vectors.

The application treats DistilBERT as a comparison model. Its result is displayed when it is available, while the final recommended category is selected using the stronger available held-out validation result.

---

## 7. Evaluation Strategy

The classifiers use a stratified train-test split with:

- Test size: 20 percent.
- Random state: `42`.
- Stratification by category.

Stratification ensures that the category distribution is represented in both training and test data. The reported validation accuracy is calculated on the held-out test portion rather than on the training set.

The interface displays each model's validation accuracy beside its prediction. This provides context for the model comparison, although validation accuracy should not be interpreted as a guarantee for an individual input.

The manual notebook also calculates category-level precision, recall, and F1 scores. These metrics are useful when the dataset is imbalanced because accuracy alone may hide poor performance on smaller categories.

---

## 8. PDF Input Workflow

The PDF workflow is implemented in `app.py` using `pypdf`.

When a PDF is uploaded:

1. The PDF reader opens the uploaded file.
2. Text is extracted page by page.
3. The page text is combined into one string.
4. A regular expression identifies the abstract section between the `Abstract` heading and likely `Keywords` or `Introduction` headings.
5. The title is selected using PDF metadata when the metadata is meaningful.
6. If metadata is unavailable or generic, title candidates are selected from the text before the abstract.
7. Author affiliations, DOI lines, URLs, year markers, and section headings are excluded from the title candidates where possible.
8. The extracted title and abstract are placed into editable interface fields for user review.

The extracted content is editable because PDF layouts differ across publishers. The extraction logic is designed for selectable text. Image-only scanned PDFs require a separate OCR pipeline and are outside the current `pypdf` text extraction path.

---

## 9. Streamlit Interface

The interface is intentionally simple so that a user can complete the workflow without interacting directly with Python code.

## 9.1 Input Selection

The user can select one of two input modes:

- Write title and abstract.
- Upload a PDF.

The abstract is optional in the interface. If no abstract is available, the system can still attempt classification from the title.

## 9.2 Classification Results

After the user selects `Classify paper`, the interface:

1. Combines the title and abstract into the classifier query.
2. Loads the cached classifier service.
3. Runs the available Word2Vec and DistilBERT prediction paths.
4. Displays the predicted category for each available model.
5. Displays category probabilities as percentages.
6. Selects the category from the strongest available model.

The classifier service is cached with Streamlit's resource cache to avoid repeatedly loading the same training artifacts during normal interaction.

## 9.3 Recommendation Results

The interface allows the user to select either three or five recommendations. The recommendation section displays:

- Paper title.
- Similarity score.
- Category.
- Citation count.
- `Top 1% cited` or `Top 10% cited` when the dataset indicates that status.
- A direct DOI or URL link when available.

If no link is present in the source metadata, the interface explicitly states that no paper link is available.

---

## 10. Recommendation Engine

The recommendation engine is designed as a two-stage retrieval process.

### Stage 1: Category Filtering

The predicted category is used to reduce the candidate set. This prevents the system from returning papers from unrelated areas when a category-specific recommendation is requested.

### Stage 2: TF-IDF Cosine Similarity

The user's abstract and each candidate paper abstract are transformed into TF-IDF vectors. The vectorizer uses English stop-word removal and unigram/bigram features.

For two vectors $x$ and $y$, cosine similarity is calculated as:

$$
\operatorname{cosine}(x,y) = \frac{x \cdot y}{\|x\|\|y\|}
$$

The result is high when the abstracts share important terms and phrase patterns. It is not automatically one merely because both papers belong to the same category. Candidate papers are sorted by similarity, and the requested Top-3 or Top-5 papers are returned.

This recommendation score measures textual similarity, not citation relationship, research quality, or factual correctness.

---

## 11. Citation and Link Metadata

The recommendation object preserves paper metadata needed for the final display.

Citation handling includes:

- Reading the `Citation count` field from the processed dataset.
- Displaying the count beside each recommended paper.
- Displaying `Top 1% cited` when that source flag is positive.
- Displaying `Top 10% cited` when the top-one-percent flag is not positive but the top-ten-percent flag is positive.
- Leaving the cited-status area blank when neither flag is positive.

For links, the service checks common link fields and the DOI field. If a DOI is present without an HTTP prefix, it is converted into a direct URL using the `https://doi.org/` format.

---

## 12. Technologies Used

### Programming Language

- Python

### User Interface

- Streamlit

### Data Processing

- CSV reader from the Python standard library.
- NumPy for numerical arrays and vector operations.
- Pandas for preprocessing and schema-aware data operations.

### Natural Language Processing

- NLTK stop words and optional WordNet lemmatization.
- Word2Vec document representations.
- Hugging Face Transformers for DistilBERT tokenization and encoding.
- Scikit-learn TF-IDF vectorization for recommendations.

### Machine Learning

- Scikit-learn `LogisticRegression`.
- Scikit-learn `StandardScaler`.
- Stratified train-test splitting.

### PDF Processing

- `pypdf` for selectable text extraction and PDF metadata access.

### Deep Learning Runtime

- PyTorch for DistilBERT inference.

---

## 13. Installation and Execution

The project uses a local virtual environment. From the repository root, install the dependencies with:

```powershell
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

If the environment already exists, run only:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Start the application with:

```powershell
.venv/Scripts/streamlit.exe run app.py
```

The default local address is:

```text
http://localhost:8501
```

The first DistilBERT query may take longer than later queries because the pretrained model must be available in the local Hugging Face cache. Word2Vec classification and TF-IDF recommendation use the saved local artifacts.

---

## 14. Command-Line Usage

The standalone DistilBERT workflow can be run with:

```powershell
.venv/Scripts/python.exe src/bert_classifier.py
```

To classify custom text:

```powershell
.venv/Scripts/python.exe src/bert_classifier.py --text "A research paper about network intrusion detection"
```

Available options include:

- `--max-rows`: number of dataset rows used for the demonstration run.
- `--batch-size`: number of texts encoded in one batch.
- `--max-length`: maximum token length for DistilBERT.
- `--cache`: location of the embedding cache file.

---

## 15. Notebooks and Experimental Work

### Word2Vec Notebook

`notebooks/word2vec.ipynb` documents the word-embedding experiment and the construction of document-level representations.

### Logistic Regression Notebook

`notebooks/logistic_regression.ipynb` demonstrates:

- Dataset loading.
- Category-aware train-test splitting.
- Feature scaling.
- Manual multiclass softmax logistic regression.
- Weighted cross-entropy training.
- Accuracy and category-level metrics.
- Custom text classification.
- DistilBERT embedding comparison.

The notebook is useful for understanding the educational implementation and the model decisions behind the reusable application service.

### CSV Merge Notebook

`utilitites/merge_csv.ipynb` documents the merging of category-specific CSV files into a combined research-paper dataset.

---

## 16. Design Decisions

### Why Use Two Representation Methods?

Word2Vec is lightweight, interpretable, and efficient for local inference. DistilBERT captures contextual language information and provides a stronger representation in many general language tasks. Keeping both approaches makes it possible to compare a classical embedding pipeline with a transformer-based pipeline.

### Why Use Logistic Regression?

Logistic regression is a strong baseline for dense document features. It is relatively simple to train, provides class probabilities, and is easy to evaluate and explain. Balanced class weights help reduce the effect of category imbalance.

### Why Use Category Filtering Before Similarity Ranking?

A system that compares an input paper with every paper in the dataset may return documents that are textually similar but belong to unrelated research areas. Category filtering combines supervised classification with content-based retrieval and keeps recommendations within the predicted research area.

### Why Use TF-IDF for Recommendations?

TF-IDF is transparent and efficient for measuring shared vocabulary and phrase importance. It is especially suitable for a recommendation layer where users benefit from an understandable similarity score. The classifier embeddings and recommendation vectors are therefore kept as separate representations for separate purposes.

---

## 17. Limitations

- PDF extraction depends on selectable text and publisher formatting.
- Metadata-based title extraction may be unavailable or generic for some PDFs.
- Image-only scanned PDFs require an OCR pipeline outside the current `pypdf` extraction path.
- The quality of DistilBERT predictions depends on the cached embeddings, model availability, and the size and balance of the embedding subset.
- Validation accuracy is a model-selection indicator, not a guarantee for individual predictions.
- Similarity scores represent textual overlap and should not be interpreted as proof of scientific equivalence.
- Recommendations are limited to papers available in the processed dataset.
- A recommendation link is available only when the source dataset contains a usable DOI or URL.
- Saved embeddings and the processed dataset must maintain the same row order.

---

## 18. Future Improvements

Potential future improvements include:

1. Add a dedicated OCR pipeline for scanned PDFs.
2. Store trained classifiers and scalers so they do not need to be retrained at application startup.
3. Generate a full-size DistilBERT embedding cache aligned with the complete dataset.
4. Evaluate additional metrics such as macro-averaged F1, confusion matrices, and per-category calibration.
5. Add hybrid recommendation scoring that combines TF-IDF similarity with contextual embedding similarity.
6. Add an explicit metadata schema validator for link, DOI, citation, and category fields.
7. Add pagination and richer paper metadata to the recommendation view.
8. Add automated tests for PDF extraction, vector alignment, classification, and recommendation ranking.
9. Add a database or search index for larger datasets.
10. Add an export option for classification and recommendation results.

---

## 19. Reproducibility and Validation

The classifier uses a fixed random seed of `42` for data splitting and deterministic subset selection. Reproducible results require the following to remain aligned:

- Processed dataset row order.
- Category labels.
- Word2Vec document vectors.
- Vocabulary and word-vector files.
- DistilBERT embedding rows.
- Python package versions.

The implementation should be validated after environment setup using:

```powershell
.venv/Scripts/python.exe -m py_compile app.py src/paper_classifier.py src/bert_classifier.py
```

The Streamlit endpoint can then be checked by opening the local URL printed by the application.

---

## 20. Conclusion

The NLP-Based Research Paper Intelligence System combines data preparation, learned document representation, supervised classification, similarity-based retrieval, and a practical user interface in one workflow. It provides a clear path from an uploaded or manually entered research paper to a predicted category and a ranked list of related papers.

The project also maintains a useful separation of responsibilities. Preprocessing prepares consistent text and metadata. Word2Vec and DistilBERT provide alternative representations. Logistic regression performs category prediction. TF-IDF cosine similarity handles transparent abstract-level retrieval. Streamlit connects these components into an accessible application.

As a research-support tool, the system is most valuable as an intelligent first-pass assistant. Its recommendations and predictions can reduce search time, while final academic judgment remains with the researcher.
