from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# -----------------------------
# Load SBERT MiniLM model
# -----------------------------
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# -----------------------------
# Function to compute similarity score
# -----------------------------
def get_similarity_score(text1, text2):
    # Tokenize and get embeddings
    def get_token_embeddings(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # [seq_len, hidden_dim]
        token_embeddings = outputs.last_hidden_state.squeeze(0)
        return token_embeddings

    emb1 = get_token_embeddings(text1)
    emb2 = get_token_embeddings(text2)

    # Compute token-to-token similarity matrix
    def compute_attention_matrix(emb1, emb2):
        emb1_norm = F.normalize(emb1, dim=1)
        emb2_norm = F.normalize(emb2, dim=1)
        attention = torch.matmul(emb1_norm, emb2_norm.T)  # [len1, len2]
        return attention

    attention_matrix = compute_attention_matrix(emb1, emb2)

    # Aggregate similarity scores: Max-pooling similarity
    max_scores_per_token = attention_matrix.max(dim=1)[0]  # max per token in sentence1
    similarity_maxpool = max_scores_per_token.mean().item()

    # Normalize to [0,1]
    similarity_maxpool = (similarity_maxpool + 1) / 2

    return similarity_maxpool


