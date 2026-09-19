# NLP-Based Research Paper Intelligence System

An end-to-end research paper classification and discovery system built with Word2Vec, DistilBERT, multiclass logistic regression, cosine similarity, and a Streamlit interface.

The system accepts either a research-paper PDF or manually entered title and abstract text. It identifies the paper's most likely research category, displays class probabilities from the available classifiers, and recommends related papers from the project dataset using abstract-level TF-IDF cosine similarity. Dataset citation information and available DOI links are shown with each recommendation.

## Project Objectives

The project addresses three related tasks:

1. Prepare and normalize a research-paper dataset for machine-learning experiments.
2. Classify papers into their research categories using learned document representations.
3. Help users discover related papers by ranking dataset documents according to textual similarity.

The current category labels are derived from the dataset's `Category` column. The available data contains categories such as `hpc`, `iot`, `networks`, `nlp`, `security`, and `vision`.

## System Features

- PDF upload through a browser-based Streamlit interface.
- Extraction of selectable PDF text, including title and abstract fields.
- Manual title input with an optional abstract.
- Word2Vec-based multiclass logistic-regression classification.
- DistilBERT embedding-based logistic-regression classification when the pretrained model is available locally or can be downloaded.
- Probability distributions for each classifier's predicted category.
- Model selection based on held-out validation accuracy.
- Top-3 or Top-5 related-paper recommendations.
- Abstract-to-abstract TF-IDF cosine similarity ranking.
- Citation count display for recommended papers.
- Optional `Top 1% cited` or `Top 10% cited` labels.
- Direct DOI or dataset-provided URL links when available.

## Architecture

```text
User
 |
 | PDF upload or title/abstract text
 v
Streamlit interface (app.py)
 |
 |-- PDF text extraction with pypdf
 |-- Input validation and presentation
 |
 v
PaperClassifier (src/paper_classifier.py)
 |
 |-- Saved Word2Vec document vectors
 |-- Saved vocabulary and word vectors
 |-- Saved DistilBERT embeddings
 |-- Logistic-regression classifiers
 |-- Category probabilities and validation scores
 |-- Abstract similarity recommendation engine
 |
 v
Processed dataset
 |
 |-- Predicted category
 |-- Ranked related papers
 |-- Similarity score
 |-- Citation count
 |-- DOI or paper link
```

### Application Layer

`app.py` is the Streamlit entry point. It manages the user workflow, accepts either manual text or a PDF, displays classifier results, and renders the related-paper list. The classifier service is cached as a Streamlit resource so the saved models and vectors are not loaded again for every interaction.

### Classification Layer

`src/paper_classifier.py` contains the reusable inference service. It loads the dataset labels and saved artifacts, trains the logistic-regression classifiers from those artifacts, creates a vector for a new input, and returns category probabilities.

The Word2Vec path uses averaged word vectors for a document. The DistilBERT path uses mean-pooled contextual embeddings from `distilbert-base-uncased`. Both paths use balanced multiclass logistic regression. The interface compares their held-out validation accuracy and uses the stronger available result as the recommended category.

### Similarity Layer

After classification, the system filters the dataset to the predicted category. It then vectorizes the user's abstract and candidate paper abstracts using TF-IDF with unigrams and bigrams. Cosine similarity is calculated between the query abstract and each candidate abstract, and the highest-scoring papers are returned.

Category membership is only used to define the candidate set. It does not make the similarity score equal to one. A score of one indicates identical normalized TF-IDF vectors; otherwise, the score reflects textual overlap between the abstracts.

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw_data/
│   │   ├── hpc.csv
│   │   ├── iot.csv
│   │   ├── networks.csv
│   │   ├── nlp.csv
│   │   ├── security.csv
│   │   └── vision.csv
│   └── processed_data/
│       ├── preprocessed_research_papers.csv
│       ├── merged_research_papers.csv
│       ├── document_vectors.npy
│       ├── document_vectors_improved.npy
│       ├── word2vec_vocabulary.csv
│       ├── word_vectors.dat
│       ├── word_vectors_improved.dat
│       ├── distilbert_embeddings.npz
│       └── distilbert_demo_embeddings.npz
├── notebooks/
│   ├── logistic_regression.ipynb
│   └── word2vec.ipynb
├── src/
│   ├── bert_classifier.py
│   ├── paper_classifier.py
│   └── preprocessing.py
└── utilitites/
	└── merge_csv.ipynb
```

## Data and Model Artifacts

### Raw Data

The `data/raw_data/` directory contains category-specific CSV files for the research-paper collection. These files are the source material for dataset merging and preprocessing.

### Processed Dataset

`preprocessed_research_papers.csv` is the main application dataset. It contains paper metadata, cleaned fields, combined text, category labels, citation metadata, and DOI information used by the interface.

Important fields include:

- `Title`
- `Abstract`
- `Category`
- `Citation count`
- `Top 1% cited`
- `Top 10% cited`
- `DOI`

### Word2Vec Artifacts

- `word2vec_vocabulary.csv` maps vocabulary terms to vector row IDs.
- `word_vectors.dat` and `word_vectors_improved.dat` contain word-level vectors.
- `document_vectors.npy` and `document_vectors_improved.npy` contain document-level vectors aligned with the processed dataset rows.

### DistilBERT Artifacts

`distilbert_embeddings.npz` stores cached contextual document embeddings. The application uses this cache to train the comparison classifier without encoding the entire dataset at every startup. The pretrained `distilbert-base-uncased` model is loaded only when a new input must be encoded.

## Preprocessing

`src/preprocessing.py` provides reusable preprocessing helpers for:

- Lowercasing and tokenization.
- Stop-word removal.
- Optional lemmatization.
- Preservation of selected technical terms such as `bert`, `cnn`, `hpc`, `gpu`, `nlp`, and `llm`.
- Construction of combined title-and-abstract text.
- Validation of the expected research-paper schema.
- Normalization of citation flags.

The preprocessing module is designed to preserve the original paper metadata while adding cleaned and combined text fields.

## Local Setup

The project uses a local Python virtual environment. From the repository root, run:

```powershell
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

If `.venv` already exists, install the requirements into that environment:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

## Run the Application

Start Streamlit from the repository root:

```powershell
.venv/Scripts/streamlit.exe run app.py
```

Then open the local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

### Using the Interface

1. Choose manual text entry or PDF upload.
2. Enter a title and optional abstract, or extract those fields from a PDF.
3. Select whether to view the Top-3 or Top-5 recommendations.
4. Click `Classify paper`.
5. Review the recommended category, classifier probabilities, similarity scores, citation counts, cited-status labels, and available paper links.

The first DistilBERT query may take longer because the pretrained model must be present in the local Hugging Face cache. Word2Vec inference and the recommendation engine use the saved project artifacts.

## Command-Line BERT Classifier

The standalone DistilBERT comparison script can be run with:

```powershell
.venv/Scripts/python.exe src/bert_classifier.py
```

To classify an individual text after training:

```powershell
.venv/Scripts/python.exe src/bert_classifier.py --text "A research paper about network intrusion detection"
```

Useful options include `--max-rows`, `--batch-size`, `--max-length`, and `--cache`.

## Notebooks

`notebooks/word2vec.ipynb` documents the Word2Vec representation workflow.

`notebooks/logistic_regression.ipynb` documents the manual multiclass softmax logistic-regression implementation, evaluation metrics, custom text prediction, and the DistilBERT comparison experiment.

`utilitites/merge_csv.ipynb` contains the CSV merging workflow used to combine raw category files.

## Technical Notes and Limitations

- PDF extraction works with selectable text. Scanned image-only PDFs require OCR and are not handled by the current `pypdf` extraction path.
- A PDF's title formatting varies by publisher. The extractor uses PDF metadata and text layout heuristics, so extracted fields should be reviewed before classification.
- DistilBERT quality depends on the cached embeddings, pretrained model availability, and the size and balance of the evaluation subset.
- Validation accuracy is a model-selection signal, not a guarantee that every individual prediction is correct.
- Cosine similarity measures textual representation overlap. It is not a citation relationship, peer-review judgment, or factual relevance guarantee.
- Recommendations can only include papers present in the processed dataset and can only provide links available in its metadata.

## Reproducibility

The classifier split uses a fixed random seed of `42`. The saved embeddings, vocabulary artifacts, processed dataset row order, and dataset labels must remain aligned for inference and recommendations to remain valid.

## License

See [LICENSE](LICENSE) for the project license.