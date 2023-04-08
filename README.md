# Fake News Detection

## Introduction
This project is a fake news detection tool that utilizes natural language processing (NLP) and machine learning (ML) techniques to classify news articles as either real or fake. The project utilizes a dataset sourced from Kaggle, which includes labeled examples of real and fake news articles.

## Technology Stack
- Python 3
- Scikit-learn
- NLTK
- Pandas
- Streamlit

## Project Structure
The project consists of the following files and directories:

- train.csv: This is the raw dataset used to train the ML model.
- Fake_News_Classifications.ipynb: This is Jupyter notebook used for exploratory data analysis and model development.
- app.py: This is the Streamlit app used to provide a front-end interface for the fake news detector.
- fake_news_model.joblib: This is the serialized ML model used to classify news articles.
- transformer_tfidf.joblib: This is the serialized TF-IDF transformer used to preprocess text data before classification.

## Installation
To install the necessary packages for this project, please run the following command:
<code> pip install -r requirements.txt

</code>

## Usage
To run the Streamlit app, please run the following command:
<code> streamlit run app.py </code>
Also, the app has been deployed on streamlit cloud, and it can be accessed through the following link: <br>

https://asmoh-fake-news-classifier-app-1x7puv.streamlit.app/ <br>

Once the app is running, you can input the title, author, and text of a news article and click "Classify" to determine if the article is real or fake.

## Conclusion
This fake news detector is a useful tool for identifying potentially false information in news articles. By leveraging NLP and ML techniques, the project is able to accurately classify articles as real or fake, allowing users to make informed decisions about the information they consume.

