import streamlit as st
import nltk
from nltk.stem import SnowballStemmer
import pickle
import sklearn


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

pnn = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

stk1 = ['i',
'me',
'my',
'myself',
'we',
'our',
'ours',
'ourselves',
'you',
"you're",
"you've",
"you'll",
"you'd",
'your',
'yours',
'yourself',
'yourselves',
'he',
'him',
'his',
'himself',
'she',
"she's",
'her',
'hers',
'herself',
'it',
"it's",
'its',
'itself',
'they',
'them',
'their',
'theirs',
'themselves',
'what',
'which',
'who',
'whom',
'this',
'that',
"that'll",
'these',
'those',
'am',
'is',
'are',
'was',
'were',
'be',
'been',
'being',
'have',
'has',
'had',
'having',
'do',
'does',
'did',
'doing',
'a',
'an',
'the',
'and',
'but',
'if',
'or',
'because',
'as',
'until',
'while',
'of',
'at',
'by',
'for',
'with',
'about',
'against',
'between',
'into',
'through',
'during',
'before',
'after',
'above',
'below',
'to',
'from',
'up',
'down',
'in',
'out',
'on',
'off',
'over',
'under',
'again',
'further',
'then',
'once',
'here',
'there',
'when',
'where',
'why',
'how',
'all',
'any',
'both',
'each',
'few',
'more',
'most',
'other',
'some',
'such',
'no',
'nor',
'not',
'only',
'own',
'same',
'so',
'than',
'too',
'very',
's',
't',
'can',
'will',
'just',
'don',
"don't",
'should',
"should've",
'now',
'd',
'll',
'm',
'o',
're',
've',
'y',
'ain',
'aren',
"aren't",
'couldn',
"couldn't",
'didn',
"didn't",
'doesn',
"doesn't",
'hadn',
"hadn't",
'hasn',
"hasn't",
'haven',
"haven't",
'isn',
"isn't",
'ma',
'mightn',
"mightn't",
'mustn',
"mustn't",
'needn',
"needn't",
'shan',
"shan't",
'shouldn',
"shouldn't",
'wasn',
"wasn't",
'weren',
"weren't",
'won',
"won't",
'wouldn',
"wouldn't",
]
# Download NLTK resources

stemmer = SnowballStemmer(language='english')

# Preprocessed Function
def Clean_text(text):
    lower_text_obj = LowerText(text)
    # Use the get_lower_text method
    text = lower_text_obj.get_lower_text()
    
    text = text.split(" ")

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)
     
    text = y[:]
    y.clear()

    for i in text:
        if i not in stk1 and i not in pnn:
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



