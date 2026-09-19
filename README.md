# NLP-Based Research Paper Intelligence System

An end-to-end natural-language-processing project for classifying research papers and finding related papers. The project combines manually implemented Word2Vec skip-gram training, document-vector classification, DistilBERT representations, a fine-tuned DistilBERT experiment, TF-IDF similarity search, and a Streamlit application.

The system accepts either a paper title and abstract or a selectable-text PDF. It predicts one of six research categories, displays class probabilities and validation accuracy, and recommends related papers from the local corpus with citation metadata and DOI links when available.

> **Implementation note:** the repository contains two DistilBERT workflows. The Streamlit application and `src/bert_classifier.py` use pretrained DistilBERT as an embedding generator followed by scikit-learn logistic regression. The notebook additionally contains an end-to-end fine-tuning experiment using `DistilBertForSequenceClassification`. These are different experiments and are documented separately below.

## Contents

- [Project objectives](#project-objectives)
- [What is implemented](#what-is-implemented)
- [System architecture](#system-architecture)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Representation learning](#representation-learning)
- [Classification models](#classification-models)
- [Recommendation method](#recommendation-method)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Running the application](#running-the-application)
- [Running the experiments](#running-the-experiments)
- [Generated artifacts](#generated-artifacts)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Reproducibility and data integrity](#reproducibility-and-data-integrity)
- [Future work](#future-work)
- [License](#license)

## Project Objectives

The project has three connected objectives:

1. Build a usable research-paper dataset from category-specific CSV files.
2. Learn numerical representations of paper text and classify papers into their research areas.
3. Help a user discover related papers by ranking abstracts according to textual similarity.

The final workflow is:

```text
Raw category CSV files
        |
        v
Merge, validate, normalize, and label
        |
        v
Processed research-paper CSV
        |
        +--> Word2Vec skip-gram --> averaged document vectors
        |                                  |
        |                                  +--> multiclass logistic regression
        |
        +--> DistilBERT encoder --> contextual document embeddings
        |                                  |
        |                                  +--> multiclass logistic regression
        |                                  |
        |                                  +--> notebook fine-tuning experiment
        |
        +--> TF-IDF abstracts --> cosine similarity recommendations
```

## What Is Implemented

- Category-specific raw CSV ingestion for HPC, IoT, networks, NLP, security, and vision.
- CSV merging and duplicate removal workflow in `utilitites/merge_csv.ipynb`.
- Schema validation and reusable text-cleaning functions in `src/preprocessing.py`.
- Manual Word2Vec skip-gram training with negative sampling in `notebooks/word2vec.ipynb`.
- 300-dimensional averaged Word2Vec document vectors.
- Manual multiclass softmax logistic regression in `notebooks/logistic_regression.ipynb`.
- Scikit-learn balanced multiclass logistic regression in the application inference service.
- DistilBERT mean-pooled embeddings with a logistic-regression classifier for the application.
- Direct DistilBERT sequence-classification fine-tuning in the notebook experiment.
- PDF title and abstract extraction using `pypdf`.
- Top-3 or Top-5 related-paper recommendations using abstract TF-IDF and cosine similarity.
- Display of category probabilities, validation accuracy, citation counts, cited-status flags, and DOI or URL links.

## System Architecture

```text
User
 |
 |-- Manual title and abstract
 |-- Selectable-text PDF
 v
Streamlit UI: app.py
 |
 |-- PDF extraction and input validation
 |-- Cached PaperClassifier instance
 v
PaperClassifier: src/paper_classifier.py
 |
 |-- Word2Vec document vectors + vocabulary + word vectors
 |-- Cached DistilBERT document embeddings
 |-- Balanced logistic-regression models
 |-- Held-out validation scores
 |-- TF-IDF recommendation engine
 v
Results
 |
 |-- Model predictions and probabilities
 |-- Best available model by validation accuracy
 |-- Related papers in the predicted category
 |-- Similarity, citation, and link metadata
```

### Application layer

`app.py` is the Streamlit entry point. It supports two input modes:

- **Write title and abstract:** the user enters text directly.
- **Upload PDF:** `pypdf.PdfReader` extracts selectable text. A regular expression locates the abstract, while PDF metadata and text-layout heuristics are used to estimate the title.

The classifier is loaded through `st.cache_resource`, so the large vector files and fitted classifiers are not reconstructed for every Streamlit interaction.

### Inference layer

`src/paper_classifier.py` loads the processed labels and saved artifacts, creates a fixed stratified 80/20 split, trains the two application classifiers, and exposes prediction and recommendation methods.

The application trains its classifiers at startup from saved representations rather than loading a serialized scikit-learn model. This keeps the representation files transparent, but increases startup time.

### Model selection

For each user input, the application obtains predictions from the available Word2Vec and DistilBERT-embedding classifiers. It selects the category from the model with the larger held-out validation accuracy. This is a simple model-selection rule, not an ensemble and not a guarantee that the selected model is best for every individual paper.

## Dataset

The dataset was collected from OpenAlex using the OpenAlex web interface and free api.

### Source files

The raw files are stored in `data/raw_data/`:

| File | Source rows | Source columns | Category |
|---|---:|---:|---|
| `hpc.csv` | 7,629 | 16 | hpc |
| `iot.csv` | 23,794 | 16 | iot |
| `networks.csv` | 7,167 | 17 | networks |
| `nlp.csv` | 11,884 | 40 | nlp |
| `security.csv` | 2,279 | 18 | security |
| `vision.csv` | 8,827 | 20 | vision |
| **Total before merging** | **61,580** |  |  |

The raw files do not all have identical schemas. The merge workflow selects the common research-paper metadata fields, assigns a category from the source filename, removes duplicate records, and writes the merged output.

### Processed dataset

The main dataset is `data/processed_data/preprocessed_research_papers.csv`.

Verified repository statistics:

- **Rows:** 46,344
- **Columns:** 18
- **Non-empty titles:** 45,687
- **Non-empty abstracts:** 45,028
- **Non-empty combined `text` values:** 46,344
- **Average abstract length:** approximately 1,463.6 characters among non-empty abstracts

The six-class distribution is:

| Category | Papers | Share |
|---|---:|---:|
| hpc | 6,663 | 14.38% |
| iot | 19,659 | 42.42% |
| networks | 5,688 | 12.28% |
| nlp | 9,137 | 19.72% |
| security | 924 | 1.99% |
| vision | 4,273 | 9.22% |
| **Total** | **46,344** | **100%** |

The large IoT class and small security class create a substantial class-imbalance problem. This is why the application classifiers use `class_weight="balanced"`, and why accuracy must be interpreted together with per-class precision, recall, and F1 score.

The processed schema is:

```text
Title
Author
Citation count
Concept
Domain
Field
Keyword
Topic
Topic IDs
Institution
Cited by
Cites
DOI
Abstract
Top 1% cited
Top 10% cited
Category
text
```

`Category` is the supervised target. `text` is the combined title and abstract string used by the embedding experiments. Citation and DOI fields are metadata for displaying recommendations; they are not classification features.

## Preprocessing

`src/preprocessing.py` provides reusable preprocessing utilities:

1. Convert missing titles and abstracts to empty strings.
2. Lowercase text.
3. Tokenize with the pattern `[a-z0-9]+(?:[-'][a-z0-9]+)*`.
4. Remove English stop words.
5. Preserve domain terms including `bert`, `cnn`, `hpc`, `gpu`, `nlp`, and `llm`.
6. Optionally lemmatize with NLTK's `WordNetLemmatizer` when the WordNet resource is available.
7. Build a combined title-and-abstract `text` field.
8. Normalize citation flags to `0` or `1`.
9. Validate the required paper metadata columns before export.

The Word2Vec notebook uses the stored `text` field and whitespace tokenization for its training corpus. The application uses a regular expression tokenizer when averaging vectors for a new query. Therefore, consistent vocabulary and artifact alignment are essential.

## Representation Learning

### Word2Vec skip-gram

The notebook implements skip-gram manually rather than calling `gensim`. Each paper is tokenized into a sequence of vocabulary IDs. For every center word, words within a window of seven positions become positive context pairs.

Configured values:

| Parameter | Value |
|---|---:|
| Vector size | 300 |
| Context window | 7 |
| Minimum count | 1 |
| Negative samples per positive pair | 2 |
| Training epochs | 5 |
| Maximum training pairs | 1,000,000 |
| Learning rate | 0.025 |
| Random seed | 42 |

The vocabulary contains one row per retained token. `W_in` stores center-word vectors and `W_out` stores context-word vectors. Both matrices are stored as float32 memory-mapped `.dat` files so that the large matrices do not need to be copied into ordinary Python memory.

### Why sigmoid is valid in skip-gram

Skip-gram with negative sampling is not the same as the project's six-class paper classifier. It creates binary pairwise decisions:

- `(center word, real context word)` has target `1`.
- `(center word, randomly sampled word)` has target `0`.

The sigmoid is therefore appropriate for each pair. Negative sampling approximates the expensive full-vocabulary softmax objective with several binary objectives.

### Document vectors

After training, the notebook represents each paper by the arithmetic mean of its known word vectors:

$$
\mathbf{d}_j = \frac{1}{|T_j|}\sum_{t \in T_j} \mathbf{v}_t
$$

where $T_j$ is the set of vocabulary tokens in document $j$ and $\mathbf{v}_t \in \mathbb{R}^{300}$. Empty documents receive a zero vector. The resulting matrix has shape $(46{,}344, 300)$ and is saved as `document_vectors.npy` or `document_vectors_improved.npy` depending on the notebook run.

### DistilBERT embeddings used by the application

The application loads `distilbert-base-uncased` through Hugging Face Transformers. For a tokenized input, the last hidden state is masked to ignore padding and mean-pooled:

$$
\mathbf{h}_{doc} = \frac{\sum_{i=1}^{L} m_i\mathbf{h}_i}{\max(\sum_{i=1}^{L}m_i,\epsilon)}
$$

Here, $\mathbf{h}_i$ is the contextual vector for token $i$, $m_i$ is the attention-mask value, and $L$ is the padded sequence length. The resulting 768-dimensional vector is passed to a logistic-regression classifier in the application.

## Classification Models

### Manual multiclass softmax logistic regression

`notebooks/logistic_regression.ipynb` contains a from-scratch classifier over the 300-dimensional Word2Vec document vectors. For input $\mathbf{x}$, the model computes one logit per category:

$$
z_k = \mathbf{x}^{T}\mathbf{w}_k + b_k
$$

The logits are converted to one probability distribution with softmax:

$$
P(y=k\mid\mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}
$$

The implementation subtracts the largest logit before exponentiation for numerical stability. The predicted category is the class with the largest probability. Unlike independent sigmoid outputs, the softmax probabilities sum to one across the six mutually exclusive categories.

The notebook uses weighted cross-entropy. For one-hot target vector $\mathbf{y}$ and probability vector $\mathbf{p}$:

$$
\mathcal{L} = -\sum_{k=1}^{K}y_k\log(p_k)
$$

The notebook computes class weights from training frequencies and applies square-root balancing:

$$
w_k = \sqrt{\frac{N}{K n_k}}
$$

where $N$ is the number of training examples, $K$ is the number of classes, and $n_k$ is the number of examples in class $k$. This gives minority categories more influence without applying the strongest possible inverse-frequency correction.

### Application Word2Vec classifier

`src/paper_classifier.py` loads the saved document vectors, fits a `StandardScaler` using only the training split, and trains scikit-learn `LogisticRegression` with:

```text
solver="lbfgs"
max_iter=1000
class_weight="balanced"
random_state=42
```

The same scaler is applied to a new averaged Word2Vec vector before prediction. The classifier returns a probability for every category with `predict_proba`.

### Application DistilBERT-embedding classifier

The application loads `distilbert_embeddings.npz`, scales the embeddings using the training portion of a stratified 80/20 split, and trains a second balanced multiclass logistic-regression model. The input text is encoded with the same pretrained DistilBERT checkpoint, mean-pooled, scaled, and classified.

This is a transfer-learning pipeline:

```text
Paper text
   -> pretrained DistilBERT encoder
   -> mean pooling
   -> StandardScaler
   -> balanced multiclass LogisticRegression
   -> category probabilities
```

The encoder is not fine-tuned in this application path. Its parameters remain fixed while logistic regression learns the category boundary.

### Notebook DistilBERT fine-tuning experiment

The notebook also contains a separate direct fine-tuning experiment using `DistilBertForSequenceClassification`:

```text
distilbert-base-uncased
   -> transformer encoder
   -> classification head with 6 logits
   -> cross-entropy loss
   -> gradient updates to encoder and head
```

The experiment uses a balanced subset of up to 600 papers, maximum sequence length 256, batch size 8, one epoch, and learning rate `2e-5`. The classification head is newly initialized for the six project categories. This explains the expected warning about newly initialized classifier weights on the first run.

For logits $\mathbf{z}$ and integer target class $y$, the training loss is:

$$
\mathcal{L}_{CE} = -\log\left(\frac{e^{z_y}}{\sum_{k=1}^{K}e^{z_k}}\right)
$$

At prediction time, softmax converts the six logits into class probabilities.

## Recommendation Method

Recommendations are generated after classification:

1. Select the category predicted by the chosen model.
2. Keep candidate papers whose `Category` matches that prediction.
3. Use the input abstract, or the title when no abstract is provided, as the query.
4. Fit a `TfidfVectorizer` over the query and candidate abstracts with English stop-word removal and unigram/bigram features.
5. Compute cosine similarity between the query vector and every candidate vector.
6. Return the three or five highest-scoring papers.

TF-IDF assigns a weight to term $t$ in document $d$ using:

$$
\mathit{tfidf}(t,d)=\mathit{tf}(t,d)\times\log\left(\frac{1+N}{1+\mathit{df}(t)}\right)+1
$$

The cosine similarity of query vector $\mathbf{q}$ and paper vector $\mathbf{p}$ is:

$$
\mathit{cos}(\mathbf{q},\mathbf{p}) = \frac{\mathbf{q}\cdot\mathbf{p}}{\|\mathbf{q}\|_2\|\mathbf{p}\|_2}
$$

Category filtering reduces the search space and improves topical focus. It does not itself increase a similarity score. A high score indicates lexical or phrase overlap in the TF-IDF representation; it is not proof of citation, methodological equivalence, or scientific quality.

## Repository Structure

```text
.
├── app.py                              # Streamlit user interface
├── requirements.txt                    # Python dependencies
├── README.md                           # This documentation
├── LICENSE
├── roadmap.txt                         # Project planning notes
├── data/
│   ├── raw_data/                       # Six source CSV files
│   └── processed_data/                 # CSVs and model artifacts
├── notebooks/
│   ├── word2vec.ipynb                  # Manual skip-gram and document vectors
│   └── logistic_regression.ipynb       # Manual classifier and DistilBERT experiment
├── src/
│   ├── preprocessing.py                # Schema and text preprocessing helpers
│   ├── paper_classifier.py             # Application inference and recommendations
│   └── bert_classifier.py               # CLI DistilBERT-embedding classifier
└── utilitites/
    └── merge_csv.ipynb                 # Raw CSV merge workflow
```

## Generated Artifacts

| Artifact | Purpose | Required by application? |
|---|---|---|
| `preprocessed_research_papers.csv` | Main labeled dataset and metadata | Yes |
| `merged_research_papers.csv` | Intermediate merged dataset | No, unless rebuilding preprocessing |
| `word2vec_vocabulary.csv` | Token-to-row mapping | Yes |
| `document_vectors_improved.npy` | Word2Vec document representations | Yes |
| `word_vectors_improved.dat` | Word-level Word2Vec vectors | Yes |
| `word_vectors_improved.dat.out` | Skip-gram output/context matrix | Needed to continue that training run; not used for inference |
| `distilbert_embeddings.npz` | Cached application embeddings | Yes for the current application path |
| `distilbert_demo_embeddings.npz` | Notebook/demo cache | No for the Streamlit application |
| `document_vectors.npy` | Earlier Word2Vec document-vector output | Used by the original notebook path; not the application default |
| `word_vectors.dat` and `.out` | Earlier Word2Vec matrix files | Used by the original notebook path; not the application default |

The binary artifacts can be regenerated, but deleting them means rerunning the relevant notebook or embedding-generation path. The CSV files are the primary data products. Do not mix vectors generated from a different dataset row order with the current labels.

## Setup

From the repository root on Windows PowerShell:

```powershell
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

If `.venv` already exists, only the installation command is needed:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

The project dependencies include pandas, NumPy, scikit-learn, PyTorch, Transformers, NLTK, `pypdf`, and Streamlit. Hugging Face may download `distilbert-base-uncased` the first time a DistilBERT path is used. An internet connection or a pre-populated local Hugging Face cache is required for that first download.

## Running the Application

Start Streamlit from the repository root:

```powershell
.venv/Scripts/streamlit.exe run app.py
```

Open the URL printed by Streamlit, normally:

```text
http://localhost:8501
```

### User workflow

1. Choose **Write title and abstract** or **Upload PDF**.
2. Enter a title and optional abstract, or extract them from a selectable-text PDF.
3. Review extracted PDF fields before classification.
4. Choose three or five recommendations.
5. Select **Classify paper**.
6. Review both model results, their held-out validation accuracy, category probabilities, and the recommended model.
7. Review related papers, cosine similarity, citation counts, cited-status flags, and available links.

The first application startup can be slow because it loads large memory-mapped Word2Vec files, trains the classifiers, and may load the DistilBERT checkpoint when a prediction is requested.

## Running the Experiments

### Merge and preprocess data

Open `utilitites/merge_csv.ipynb` and run its cells to recreate the merged dataset. The preprocessing helpers in `src/preprocessing.py` can then validate and export the project schema. Because the raw files have different column sets, inspect the required-column validation errors if a new source file is added.

### Train Word2Vec

Open `notebooks/word2vec.ipynb` and run the cells in order. The notebook:

1. Loads `preprocessed_research_papers.csv`.
2. Builds a vocabulary and integer token sequences.
3. Generates up to one million center-context pairs.
4. Trains skip-gram with two negative samples per positive pair.
5. Saves word matrices as memory-mapped `.dat` files.
6. Averages word vectors into one 300-dimensional vector per paper.
7. Saves document vectors and an embeddings-enriched CSV.
8. Verifies dimensions and finite values.

### Evaluate logistic regression

Open `notebooks/logistic_regression.ipynb` and run the manual Word2Vec section in order. It performs a stratified 80/20 split, standardizes using training data only, trains a six-class softmax model, reports accuracy and per-class precision/recall/F1, and accepts custom text for prediction.

The same notebook includes the direct DistilBERT fine-tuning experiment. It is computationally heavier than the Word2Vec classifier, especially on CPU. The notebook experiment is separate from the application, which currently uses frozen DistilBERT embeddings plus logistic regression.

### Run the CLI embedding classifier

```powershell
.venv/Scripts/python.exe src/bert_classifier.py
```

Classify a supplied text after training:

```powershell
.venv/Scripts/python.exe src/bert_classifier.py --text "A research paper about network intrusion detection"
```

Useful options are `--max-rows`, `--batch-size`, `--max-length`, and `--cache`.

## Evaluation

The project reports:

- **Accuracy:** proportion of correctly classified papers.
- **Precision:** among papers predicted as a class, the proportion that truly belongs to that class.
- **Recall:** among papers belonging to a class, the proportion recovered by the model.
- **F1 score:** harmonic mean of precision and recall.
- **Cosine similarity:** normalized vector similarity used only for recommendation ranking.

The manual Word2Vec classifier is evaluated on a stratified held-out split. The notebook DistilBERT fine-tuning experiment uses a balanced subset of up to 600 papers and a separate stratified test split. The application trains its two logistic-regression models on the full processed representation matrix using a fixed stratified 80/20 split and displays each held-out accuracy.

Reported results are experiment outputs, not universal benchmarks. Results can change if the processed row order, cached artifact version, random seed, tokenizer, model checkpoint, or training subset changes.

## Limitations

### Dataset limitations

- The six categories are inherited from source-file labels and may not represent the full research-paper taxonomy.
- The class distribution is highly imbalanced, especially for security and IoT.
- Duplicate removal and source-specific schemas can affect which papers survive preprocessing.
- Metadata quality, citation counts, DOI values, and cited-status flags depend on the source CSV files.
- The repository contains a large local dataset and binary artifacts; cloning and storage requirements are substantial.

### Text and PDF limitations

- `pypdf` extracts selectable text but does not perform OCR. Scanned image-only PDFs are unsupported.
- PDF layouts vary widely. Title and abstract extraction is heuristic and should be reviewed by the user.
- Very long papers are truncated by the DistilBERT tokenizer at 256 tokens in the current implementation. Important information after the truncation point is not represented.
- Empty or extremely short inputs can produce weak vectors or no known Word2Vec tokens.
- Word2Vec averaging removes word order and compresses an entire paper into one vector.

### Modeling limitations

- The Word2Vec implementation uses a capped set of training pairs and only five epochs; it is an educational implementation rather than a state-of-the-art embedding model.
- Negative sampling uses randomly selected negative IDs and does not implement a frequency-smoothed sampling distribution.
- The application uses frozen DistilBERT embeddings with a linear classifier. It does not fine-tune DistilBERT for the project labels.
- The notebook fine-tuning experiment uses only one epoch and a small balanced subset, so it should not be treated as a fully optimized production model.
- Validation accuracy can favor the majority class and should be read with macro and per-class metrics.
- The model-selection rule chooses the classifier with higher aggregate held-out accuracy; it does not calibrate probabilities or combine models.
- No serialized trained classifier, experiment registry, hyperparameter search, or cross-validation pipeline is currently included.

### Recommendation limitations

- Recommendations are restricted to papers already present in the processed CSV.
- The system ranks lexical TF-IDF overlap, not semantic equivalence, citation influence, novelty, or research quality.
- Filtering candidates by predicted category can exclude relevant papers whose source label is different or whose category was predicted incorrectly.
- Missing abstracts, citation metadata, or DOI links reduce recommendation quality and displayed information.

## Reproducibility and Data Integrity

The main split uses `random_state=42` and stratification. Reproducible results additionally require:

1. The same processed CSV and category labels.
2. The same row ordering between labels and saved document vectors.
3. The matching vocabulary and word-vector matrix.
4. The same DistilBERT checkpoint and tokenizer.
5. The same dependency versions and numerical environment.

The `.npy`, `.dat`, and `.npz` files are caches and model representations, not independent datasets. They can be deleted and regenerated, but they must be regenerated from the same processed data before inference. The `.dat.out` files are output/context matrices from skip-gram training and are not required by the application's inference path.

## Future Work

Potential improvements include:

- Add OCR for scanned PDFs.
- Fine-tune and save a production DistilBERT classifier rather than using frozen embeddings in the application.
- Use cross-validation and macro-F1 for model selection under class imbalance.
- Tune the Word2Vec negative-sampling distribution and train for more epochs.
- Add confusion matrices and error analysis by category.
- Add approximate nearest-neighbor search for faster recommendations on larger corpora.
- Use semantic sentence embeddings or a cross-encoder reranker for recommendations.
- Persist fitted scalers and classifiers to avoid retraining at every application startup.
- Add automated tests for preprocessing, PDF extraction, artifact alignment, classification, and recommendation ranking.
- Track dataset versions, model parameters, and evaluation results in an experiment manifest.

## License

See [LICENSE](LICENSE) for the project license.
