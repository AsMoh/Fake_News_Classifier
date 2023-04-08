import streamlit as st
import pandas as pd
import numpy as np
import joblib
from nltk.stem import PorterStemmer
import re
import contractions
import nltk

nltk.download('punkt')

# Load the classification model
model = joblib.load('fake_news_model.joblib')
transformer = joblib.load('transformer_tfidf.joblib')


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
    st.sidebar.markdown("# Fake News Classifier")
    st.sidebar.markdown("Enter the title, author, and text of the news article you want to check for authenticity.")
    st.sidebar.markdown("The app will analyze the input and predict whether the news is real or fake.")

    st.title("Fake News Classifier")

    # Add text input fields for title, author and news text
    st.subheader("Enter the News Article Details")
    title = st.text_input("Title")
    author = st.text_input("Author")
    text = st.text_area("Text", height=100)

    # Add a button to classify the news
    if st.button("Check for Fake News"):
        # Make a prediction using the classification model
        df = pd.DataFrame([[title, author, text]], columns=["title", "author", "text"])
        df.fillna("", inplace=True)
        df['content']= df['title']+" "+df['author']+" "+df['text']
        df['content']=clean_input(df['content'])
        
        transformed_txt=transformer.transform(df['content'])
        prediction = model.predict(transformed_txt)
        
        # Display the prediction
        if prediction == 1:
            st.error("This news is likely fake!")
        else:
            st.success("This news is likely real.")

    # Add a footer with some information about the app
    st.markdown("---")
   
if __name__ == "__main__":
    app()
    
