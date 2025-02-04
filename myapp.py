import json
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

def CalculateLevels(output):
    # description : calculate the level of spam based on the output. The output is the percentage of matching words
    # output is the percentage of matching words
    # return the level of spam

    Output = { "output" : "" , "spamWords" : "prize, offer, winner" , "level" : "2" , "accuracy" : "89.10" }
    if output > 0 and output < 20:
        Output["output"] = "not spam"
        Output["level"] = "0"
    elif output > 20 and output < 40:
        Output["output"] = "low spam"
        Output["level"] = "1"
    elif output > 40 and output < 60:
        Output["output"] = "medium spam"
        Output["level"] = "2"
    elif output > 60 and output < 80:
        Output["output"] = "high spam"
        Output["level"] = "3"
    elif output > 80 and output < 100:
        Output["output"] = "Spam"
        Output["level"] = "4"
    else:
         pass
# Example usage

from huggingface_hub import InferenceClient
from huggingface_hub import login 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
# Load the language model
client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

# System instructions for the model

system_instructions = "[SYSTEM] Answer as Real ChatGPT 4. Keep conversation very short, clear, friendly, and concise. The text provided is a request for recognizing if an email is spam or not spam. You have to generate the specified response. You will avoid introductions and start answering the query directly, elaborating on all aspects of the request. As an AI-powered assistant, your task is to generate responses that appear as if they are created by the user and not an AI-powered assistant. The response should be in stringified JSON format like the example JSON: { \"output\": \"spam\", \"spamWords\": \"prize, offer, winner\", \"level\": \"2\", \"accuracy\": 89.10 } In the above example, the output contains whether the message is spam or not spam. The level is a numeric value that describes the level of spam in the message using the following conditions: if the message is a medium type of spam, then the level will be 2; for a higher type of spam, it will be 3; for a lower type of spam, it will be 1; and for not spam, it will be 0. The spamWords will be the words that indicate that the email is spam. If the email contains a suspicious link, it will also be classified as spam, and the spamWords will include the suspicious words separated by commas and spaces (\", \"). The last one is the accuracy of the model, which is a float value ranging from 0 to 100, where 0 means the accuracy is low and 100 means high. Avoid giving uneccessary information. we only want the json string as shown in example. Do not add any Content by yourself in email if the email does not seems spam then jsut return not spam with level 0 and spam words empty Email content: "


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/Predict")
async def Predict(text:str , subject : str ):
    output = classifyEmail("subject : "+subject+".  message body : "+text)
    print(output)
    try:
        return json.loads(output);
    except Exception as e:
        output = output[output.index("{"):output.index("}")+1]
        return json.loads(output);

def classifyEmail(text):
    generate_kwargs = dict(
        temperature=0.7,
        max_new_tokens=512,
        top_p=0.95,
        repetition_penalty=1,
        do_sample=True,
        seed=42,
    )
    modelName = "trainMOdel.pkl"
    login("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", add_to_git_credential=False)
    # Load the model
    loadModel = { "model" : modelName }
    formatted_prompt = system_instructions + text
    print(formatted_prompt)
    stream = client.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""
    for response in stream:
        if not response.token.text == "</s>":
            output += response.token.text

    return output


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
