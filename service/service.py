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
        token_embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
        return token_embeddings

    emb1 = get_token_embeddings(text1)
    emb2 = get_token_embeddings(text2)

    # Compute token-to-token similarity matrix
    emb1_norm = F.normalize(emb1, dim=1)
    emb2_norm = F.normalize(emb2, dim=1)
    attention_matrix = torch.matmul(emb1_norm, emb2_norm.T)  # [len1, len2]

    # Max similarity per token
    max_scores_per_token = attention_matrix.max(dim=1)[0]  # [len1]

    # Softmax-weighted pooling
    weights = F.softmax(max_scores_per_token, dim=0)      # importance weights
    similarity_weighted = (max_scores_per_token * weights).sum()

    # -----------------------------
    # Sentence-level similarity
    # -----------------------------
    sent_emb1 = emb1.mean(dim=0)
    sent_emb2 = emb2.mean(dim=0)
    sent_score = F.cosine_similarity(sent_emb1, sent_emb2, dim=0)

    # Combine token-level and sentence-level similarity
    combined_score = (similarity_weighted + sent_score) / 2

    # Normalize to [0,1]
    combined_score = ((combined_score + 1) / 2).item()

    return combined_score
