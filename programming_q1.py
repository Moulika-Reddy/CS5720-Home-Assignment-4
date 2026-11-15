
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download the required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')


# Function to preprocess a sentence
def preprocess_text(sentence):

    # Step 1: Tokenize the sentence into words and punctuation
    tokens = word_tokenize(sentence)
    print("Original Tokens:", tokens)

    # Step 2: Remove English stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [
        word for word in tokens
        if word.lower() not in stop_words
    ]
    print("Tokens Without Stopwords:", filtered_words)

    # Step 3: Apply stemming to the remaining words
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    print("Stemmed Words:", stemmed_words)


# Input sentence provided
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."

# Call the preprocessing function
preprocess_text(sentence)
