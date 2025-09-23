from sentence_transformers import CrossEncoder
from utils import remove_stopwords

# Pre-trained Cross-Encoder for STS (semantic textual similarity)
model = CrossEncoder('cross-encoder/stsb-roberta-large')  # or any smaller one like stsb-distilroberta-base

def get_similarity_score(sentence1: str, sentence2: str) -> float:
    """Compute similarity score between two sentences after removing stopwords."""
    # Preprocess sentences by removing stopwords
    processed_sentence1 = remove_stopwords(sentence1)
    processed_sentence2 = remove_stopwords(sentence2)
    
    # Get similarity score
    score = model.predict([[processed_sentence1, processed_sentence2]])
    return float(score[0])