import joblib
import pandas as pd
import nltk 
import fastapi 

from collections import Counter


from nltk.corpus import stopwords    

import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from sklearn.svm import SVC

from wordcloud import WordCloud    

# Downloading NLTK data
nltk.download('stopwords')    
nltk.download('punkt')        



from sklearn.preprocessing import LabelEncoder
# Importing the Porter Stemmer for text stemming
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer

# Importing the string module for handling special characters
import string

# Creating an instance of the Porter Stemmer
ps = PorterStemmer()

# Lowercase transformation and text preprocessing function
def transform_text(text):
    # Transform the text to lowercase
    text = text.lower()
    
    # Tokenization using NLTK
    text = nltk.word_tokenize(text)
    
    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    # Removing stop words and punctuation
    text = y[:]
    y.clear()
    
    # Loop through the tokens and remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
        
    # Stemming using Porter Stemmer
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    
    # Join the processed tokens back into a single string
    return " ".join(y)

import pandas as pd

def calculate_matching_percentage(sentence, dataframe):
    """
    Calculate the percentage of matching words between a sentence and a DataFrame with weights.

    Args:
        sentence (str): The input sentence.
        dataframe (pd.DataFrame): The DataFrame with columns '0' for words and '1' for weights.

    Returns:
        float: The percentage of matching words based on weights.
    """
    # Tokenize the sentence into words
    sentence_words = sentence.lower().split()

    # Calculate the percentage of matching words with weights
    total_weight = dataframe[1].sum()
    matching_weight = sum(dataframe[dataframe[0].isin(sentence_words)][1])

    percentage_matching = (matching_weight / total_weight) * 100
    if percentage_matching > 5 and percentage_matching < 50 :
        percentage_matching+= 20
    return percentage_matching

# Example usage

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/Predict")
async def Predict(text:str , subject : str , actualOutput:str):
    actualOutput = actualOutput.lower()
    df = pd.read_csv('spam.csv', encoding='latin1')
    df.columns  = ['target', 'text']
    encoder = LabelEncoder()
    text = subject+" "+text 
    print(text)
    df = pd.concat([pd.DataFrame([{'target': actualOutput, 'text': text}]),df])
    df['target'] = encoder.fit_transform(df['target'])
    df = df.drop_duplicates(keep = 'first')
    df['num_characters'] = df['text'].apply(len)
    df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
    df['num_sentence'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
    df['transformed_text'] = df['text'].apply(transform_text)
    spam_carpos = []
    for sentence in df[df['target'] == 1]['transformed_text'].tolist():
        for word in sentence.split():
            spam_carpos.append(word)
    filter_df = pd.DataFrame(Counter(spam_carpos).most_common(20))

    tfid = TfidfVectorizer(max_features = 3000)
    X = tfid.fit_transform(df['transformed_text']).toarray()
    y = df['target'].values
    testX = X[0]
    testY = y[0]
    X = X[1:]
    y = y[1:]
    svc = SVC(kernel= "sigmoid", gamma  = 1.0)
    svc.fit(X,y)
    y_pred = svc.predict([testX])
    accuracy = accuracy_score([y[0]], y_pred)
    print(accuracy)
    output = "Not Spam"
    level = 0
    percentage = 0
    if y_pred[0]== 1:
        output = "Spam"
        text = transform_text(text)
        percentage = calculate_matching_percentage(text , filter_df)
        if  percentage >1 and percentage < 25:
            level= 1
        if  percentage >26 and percentage < 60:
            level= 2
        if  percentage >61 and percentage < 100:
            level= 3
    print({ "output":output , "percentage":percentage , "level":level , "accuracy":accuracy})
    return { "output":output , "percentage":percentage , "level":level , "accuracy":accuracy}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)