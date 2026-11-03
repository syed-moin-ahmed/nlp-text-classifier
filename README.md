# NLP Resume Classifier

Classifies a resume into one of 24 job categories using natural language processing and machine learning. Paste resume text or upload a PDF, and the app returns the predicted category with its top-3 predictions.

**Live app:** https://nlp-text-classifier-z6eyltgjchj8v2ps7xpck5.streamlit.app/

## How it works

1. **Clean the text:** tokenization, stop word removal and lemmatization with NLTK
2. **Extract features:** convert text into TF-IDF vectors
3. **Classify:** predict the job category with a linear SVM
4. **Show results:** display the prediction and top-3 categories in a Streamlit app

## Results

Models were trained on 1,987 resumes (1,589 training, 398 test) across 24 categories.

| Model | Test accuracy |
|-------|---------------|
| **Linear SVM** | **61.3%** |
| Logistic Regression | 59.8% |
| Naive Bayes | 52.5% |

Training code is in [notebooks/model_training.ipynb](notebooks/model_training.ipynb). The dataset is not included in this repo.

## Tools

Python, scikit-learn, NLTK, pandas, NumPy, Streamlit, Plotly, pdfplumber, joblib

## Run locally

```bash
git clone https://github.com/syed-moin-ahmed/nlp-text-classifier.git
cd nlp-text-classifier
pip install -r requirements.txt
streamlit run app.py
```

## Next steps

Try a larger dataset and transformer models such as BERT to improve accuracy.

## Author

Syed Moin Ahmed: [LinkedIn](https://www.linkedin.com/in/syed-moin-ahmed-710812325/) | [GitHub](https://github.com/syed-moin-ahmed)
