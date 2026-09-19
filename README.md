# NLP-Based-Research-Paper-Intelligence-System

## Dataset report

Generate a Markdown quality report for the merged dataset:

```bash
python src/dataset_report.py
```

The report is saved to `results/dataset_report.md` and includes schema checks, empty values, duplicate rows and titles, abstract-length statistics, and distributions for `Topic`, `Domain`, and `Field`.

## Paper intelligence interface

Create the local environment and install the interface dependencies:

```bash
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Start the Streamlit application:

```bash
.venv/Scripts/streamlit.exe run app.py
```

Open the local URL shown by Streamlit. The interface accepts a PDF or manually entered title and abstract, extracts selectable PDF text, and compares the saved Word2Vec and DistilBERT classifiers. The model with the higher held-out validation accuracy is shown as the recommended result.