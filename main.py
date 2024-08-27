import streamlit as st
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string
import pickle


class LowerText:
    def __init__(self, text):
        self.original_text = text
        self.lower_text = text.lower()

    def get_lower_text(self):
        return self.lower_text

    
# Load the TF-IDF vectorizer (make sure it's also downloaded similarly if needed)
try:
    
    punn = pickle.load(open('punctuation.pkl', 'rb'))
    stk = pickle.load(open('stopwords.pkl', 'rb'))
    stemmer = pickle.load(open('SnowballStemmer.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading pickles: {e}")
    st.stop()

# Download NLTK resources
punn = string.punctuation
stk = stopwords.words('english')
stemmer = SnowballStemmer(language='english')

# Preprocessed Function
def Clean_text(text):
    lower_text_obj = LowerText(text)
    # Use the get_lower_text method
    text = lower_text_obj.get_lower_text()
    
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)
     
    text = y[:]
    y.clear()

    for i in text:
        if i not in stk and i not in punn:
            y.append(i)        

    text = y[:]
    y.clear()
    
    for i in text:
        y.append(stemmer.stem(i))

    return " ".join(y)

st.title("Welcome to Emotion Detector From Text Model")
st.text("Here the Model can predict the Emotions from the given Text")

text = st.text_area("Enter Text Here")

if st.button("Done", type="primary"):

    transformed_text = Clean_text(text)

    vector_input = tfidf.transform([transformed_text])

    # Predict directly without additional scaling
    result = model.predict(vector_input)[0]

    if result == 0:
        st.write("Sad 🥺")
    elif result == 1:
        st.write("Joy 😂")  
    elif result == 2:
        st.write("Love 🥰")  
    elif result == 3:
        st.write("Anger 😠")  
    elif result == 4:
        st.write("Fear 😨")
    else:
        st.write("No Emotion Found From the Text")    



