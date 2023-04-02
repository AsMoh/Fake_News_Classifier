import streamlit as st
import pandas as pd
import numpy as np
import joblib
from nltk.stem import PorterStemmer
import re
import contractions
import nltk

# Load the classification model
model = joblib.load(r'C:\Users\Asia\Data_Science_Code\new intership\Neural Networks\NLP\Assignment-fake_news\fake_news_model.joblib')
transformer = joblib.load(r'C:\Users\Asia\Data_Science_Code\new intership\Neural Networks\NLP\Assignment-fake_news\transformer_tfidf.joblib')

stemmer=PorterStemmer()

stop_wrds=['sometimes', 'ever', 'already', 'that', 'else', 'each', 'then', 'itself', 'whom', 'quite', "should've", 'regarding']

def clean_input(df):

    df= df.str.lower().str.strip()
    df =df.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df= df.apply(lambda x: contractions.fix(x))
    df= df.apply(nltk.word_tokenize)

    df= df.apply( lambda x: [ word for word in x if word not in stop_wrds])
    df= df.apply( lambda x: [ stemmer.stem(w) for w in x])
    df= df.apply( lambda x: " ".join(x) )
    return df
    
    
# Define the app
def app():
    st.set_page_config(page_title="Fake News Classifier", page_icon=":guardsman:", layout="wide")

    # Define the layout of the app
    st.title("Fake News Classifier")
    col1, col2, col3 = st.columns(3)

    # Add text input fields for title, author and news text
    with col1:
        st.markdown("<h3 style='text-align: center; font-weight: bold;'>Title</h3>", unsafe_allow_html=True)
        title = st.text_input(" " , key="title")
    with col2:
        st.markdown("<h3 style='text-align: center; font-weight: bold;'>Author</h3>", unsafe_allow_html=True)
        author = st.text_input(" " , key="author")
    with col3:
        st.markdown("<h3 style='text-align: center; font-weight: bold;'>News Text</h3>", unsafe_allow_html=True)
        text = st.text_area(" ", key="text",height=100)

    # Add a button to classify the news
    if st.button("Classify"):
        # Make a prediction using the classification model
        df = pd.DataFrame([[title, author, text]], columns=["title", "author", "text"])
        df.fillna("", inplace=True)
        df['content']= df['title']+" "+df['author']+" "+df['text']
        df['content']=clean_input(df['content'])
        
        transformed_txt=transformer.transform(df['content'])
        prediction = model.predict(transformed_txt)
        print(prediction , type(prediction))
        # Display the prediction
        if prediction == 1:
            st.error("This news is likely fake!")
        else:
            st.success("This news is likely real.")
app()
