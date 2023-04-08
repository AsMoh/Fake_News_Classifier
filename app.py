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
stop_wrds=["'m", 'was', 'below', 'rather', 'although', 'is', 'him', 'one', 'that', 'y', 'describe', 'elsewhere', 'show', 
           'whenever', 'sixty', "needn't", 'whoever', 'seeming', 'they', 'too', '‘s', 'same', 'everyone', 'all', 
           'please', 'wasnt', 'be', 'various', 'with', 'last', 'really', 'keep', "hadn't", 'wouldnt', 'maynt', 't', 's', 'regarding', 'been', 'what', 'via', 'fill', 'each', 'whither', 'hasn', 'wherever', 'dont', 'full', '‘ve', 'isnt', "she's", 'three', 'throughout', "mustn't", 'yet', 'ten', 'wasn', 'weren', 'con', 'during', 'get', 'haven', 'see', 'becoming', 'himself', 'whole', 'am', 'hereafter', 'otherwise', 'beyond', 'meanwhile', 'the', 'seemed', 'behind', 'doesn', 'go', 've', 'aren', 'ever', 'moreover', 'became', 'hadnt', 'indeed', 'therein', 'per', 'become', 'afterwards', 'here', 'mostly', 'together', 'cannot', 'anyhow', 'among', 'no', 'fire', 'whom', 'ltd', 'almost', "that'll", 'side', 'down', 'latter', 'having', 'take', 'shan', 'ourselves', 'because', 'whatever', 'toward', 'don', 'mightnt',
           'beside', 'many', 'beforehand', 'kg', 'would', 'ours', 'even', 'upon', 'hadn', 'mill', 'mightn', 'somewhere', 'isn', 'did',
           'hundred', 'say', 'hence', 'everywhere', 'arent', 'while', 'yours', 'may', 'hasnt', 'on', 'us', 'km',
           "'re", '‘m', 'whether', 'none', 'using', "it's", 'third', 'since', 'have', 'into', 'which', 'must', 
           "wouldn't", 'seems', 'anywhere', 'such', 'where', 'of', 'didnt', 'across', "you're", 'those', 'nowhere', 'someone', 'still', 'never', 'interest', "didn't", 'whence', "shan't", 'had', 'any', "'ll", 'couldnt', "'d", 'shouldn', 'o', 'it', 'her', '’d', 'thereupon', '‘d', 'you', 'amoungst', 'thence', 'than', 'bottom', 'thru', 'towards', 'ma', '’s', 'from', 'etc', 'co', 'if', 'me', 'whereas', 'only', 'amount', 'doesnt', 'now', 'yourself', 'could', 'sincere', 'about', 'front', 'except', 'five', 'out', 'found', 're', 'top', "haven't", 'werent', 'six', 'to', 'can', '‘re', 'another', 'herself', 'some', "shouldn't", 'other', 'when', 'ie', '’ve', 'these', "mightn't", 'next', 'nobody', 'quite', 'twenty', 'she', 'we', 'whereby', 'should', 'for', 'how', 'do', 'nothing', 'sometimes', 'un', 'cry', "isn't", 'own', 'but', 'not', 'were', 'm', 'doing', 'whose', 'most', 'he', "'ve", 'besides', 'our', 'wouldn', 'his', 'namely', 'n’t', 'done', 'at', 'much', 'empty', "won't", 'de', 'seem', 'find', 'once', 'others', 'nevertheless', "doesn't", "'s", 'thereby', "should've", 'mine', 'under', 'will', 'might', 'who', 'fifteen', 'whereupon', 'within', "you've", 'noone', 'used', 'system', 'made', 'however', 'over', 'their', "wasn't", "hasn't", 'thin', '’ll', 'against', 'part', 'its', 'thick', 'also', 'as', 'nor', 'something', 'well', 'twelve', 'didn', 'thereafter', 'less', 'ain', 'has', 'somehow', 'therefore', 'why', 'everything', 'hereby', 'else', 'always', '’m', "n't", 'eight', 'll', "you'll", 'further', 'often', 'serious', 'mustn', 'again', 'first', 'hereupon', "aren't", 'there', 'then', 'alone', 'my', 'eg', 'off', 'onto', 'in', 'back', 'couldn', 'your', 'ca', 'every', 'n‘t', 'anyone', 'computer', 'needn', 'more', '‘ll', 'does', 'hers', 'call', 'detail', 'becomes', 'by', 'without', 'sometime', 'just', 'either', 'so', 'anyway', 'around', 'whereafter', 'between', 'nine', 'yourselves', 'through', 'or', 'due', 'theirs', "couldn't", 'this', 'several', 'and', 'themselves', 'few', 'won', 'give', '’re', 'fifty', 'after', "you'd", 'cant', 'herein', 'before', 'both', 'put', 'i', 'formerly', 'unless', 'move', 'havent', 'though', 'inc', 'an', 'shouldnt', 'amongst', 'name', 'up', "weren't", 'myself', 'former', 'itself', 'eleven', 'four', 'enough', 'along', 'wherein', 'very', 'thus', 'neither', 'are', 'bill', 'already', 'wont', 'least', 'shallnt', 'two', 'latterly', 'being', 'anything',
           'above', 'make', 'd', 'until', 'perhaps', "don't", 'a', 'them', 'forty']

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
    
