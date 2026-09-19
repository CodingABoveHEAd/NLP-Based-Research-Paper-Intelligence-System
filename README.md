# NLP-Based-Research-Paper-Intelligence-System

## Dataset report

Generate a Markdown quality report for the merged dataset:

```bash
python src/dataset_report.py
```

The report is saved to `results/dataset_report.md` and includes schema checks, empty values, duplicate rows and titles, abstract-length statistics, and distributions for `Topic`, `Domain`, and `Field`.

## Prince analysis interface

Run the integrated interface from the project root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app accepts the processed CSV from the sidebar and provides:

- Word2Vec and BERT Logistic Regression classification notebooks
- Accuracy, precision, recall, and F1-score evaluation
- Token-frequency keywords and year/category publication trends
- Rare and recent bigrams as potential research directions for human review
- Text classification with both trained models