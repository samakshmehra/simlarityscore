import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

def remove_stopwords(text):
    """Remove stopwords from input text."""
    # Tokenize the text
    words = word_tokenize(text.lower())
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    return " ".join(filtered_words)

