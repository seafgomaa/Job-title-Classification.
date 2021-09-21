from flask import Flask, jsonify, request
import numpy as np
import pickle as pk
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pk

def count_punctuations(text):
    punctuations='''!"#$%&()'*+,-./:;<=>?@[\]^_`{|}~'''
    d=dict()
    for i in punctuations:
        d[str(i)+' count']=text.count(i)
    return d 


def count_stopwords(text):
    stop_words = set(stopwords.words('english'))  
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)

def remove_punc(txt):
    punctuations='''!"#$%&()'*+,-./:;<=>?@[\]^_`{|}~'''
    for i in punctuations:
        txt=txt.replace(i,' ')
    return txt

def remove_stopwords(txt):
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(txt)
    stopwords_x = [w for w in word_tokens if w not in stop_words]        
    return ' '.join(stopwords_x)



def adding_features(df,txt):
    df['punct_count'] = df[txt].apply(lambda x:count_punctuations(x))
    df_punct = pd.DataFrame(list(df.punct_count))
    df = pd.merge(df, df_punct, left_index=True, right_index=True)
    df.drop(columns=['punct_count'],inplace=True)
    
    df['job title'] = df['job title'].apply(lambda x:remove_punc(x))
    
    df['stopword_count'] = df[txt].apply(lambda x:count_stopwords(x))
    df['job title']=df['job title'].apply(lambda x: remove_stopwords(x))
    
    return df


def pre_process_input(txt):
    txt=txt.lower()
    arr=[txt]
    arr=[txt]
    entry=pd.DataFrame(arr, columns=['job title'])
    final_txt=adding_features(entry,'job title')
    vectorizer=pk.load(open('vectorizer', 'rb'))
    idf_features=vectorizer.transform(final_txt['job title']).toarray()
    idf= pd.DataFrame(idf_features)    
    final_txt = pd.merge(idf,final_txt,left_index=True, right_index=True)
    final_txt=final_txt.drop(['job title'],axis=1)
    return final_txt


        



app = Flask(__name__)

model = pk.load(open('SGDClassifier', 'rb'))


@app.route('/', methods=['POST'])
def addOne():
    txt=request.data
    final_txt=pre_process_input(str(txt))
    predict=model.predict(final_txt)[0]
    return predict

if __name__ == '__main__':
	app.run(debug=True, port=8080) 
