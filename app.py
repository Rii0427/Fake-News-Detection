import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re  #for regex check
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
vec = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl','rb'))
model_form = pickle.load(open('model.pkl','rb'))

def stemming(text):
    retext = re.sub('[^a-zA-Z]',' ',text)
    retext=retext.lower()
    retext=retext.split()
    retext=[port_stem.stem(word) for word in retext if not word in stopwords.words('english')]
    retext=''.join(retext)
    return retext

def check_news(news):
    news = stemming(news)
    input_data = [news]
    vector = vector_form.transform(input_data)
    pred = model_form.predict(vector)
    return pred

if __name__ == '__main__':
    st.title("Fake News Classification App")
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "Some news", height=200)
    predict = st.button("Predict")
    if predict:
        pred = check_news(sentence)
        print(pred)
        if(pred==[0]):
            st.success('Real News')
        else:
            st.warning('Fake News')