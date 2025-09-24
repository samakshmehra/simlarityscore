from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load SBERT MiniLM model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to compute similarity score
def get_similarity_score(text1, text2):
    # Tokenize both sentences in one batch
    inputs = tokenizer([text1, text2], return_tensors="pt", padding=True, truncation=True, max_length=512)
    attention_mask = inputs['attention_mask']  # [2, seq_len]

    # Forward pass for both sentences at once
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state  # [2, seq_len, hidden_dim]

    # Separate embeddings
    emb1_tokens = token_embeddings[0]  # [seq_len1, hidden_dim]
    emb2_tokens = token_embeddings[1]  # [seq_len2, hidden_dim]
    mask1 = attention_mask[0].unsqueeze(-1)   # [seq_len1, 1]
    mask2 = attention_mask[1].unsqueeze(-1)   # [seq_len2, 1]

    # Mean-pool sentence embeddings with attention mask
    sent_emb1 = (emb1_tokens * mask1).sum(dim=0) / mask1.sum()
    sent_emb2 = (emb2_tokens * mask2).sum(dim=0) / mask2.sum()

    # Normalize token embeddings
    emb1_norm = F.normalize(emb1_tokens, dim=1)
    emb2_norm = F.normalize(emb2_tokens, dim=1)

    # Token-to-token similarity matrix (both directions)
    sim_1_to_2 = emb1_norm @ emb2_norm.T  # [len1, len2]
    sim_2_to_1 = emb2_norm @ emb1_norm.T  # [len2, len1]

    # Max similarity per token in both directions
    max_scores_1 = sim_1_to_2.max(dim=1)[0]
    max_scores_2 = sim_2_to_1.max(dim=1)[0]

    # Symmetric greedy matching: average both directions
    max_scores_tokens = torch.cat([max_scores_1, max_scores_2], dim=0)

    # Softmax-weighted pooling over token-level max similarities
    weights = F.softmax(max_scores_tokens, dim=0)
    similarity_weighted = (max_scores_tokens * weights).sum()

    # Sentence-level similarity
    sent_score = F.cosine_similarity(sent_emb1, sent_emb2, dim=0)

    # Combine token-level and sentence-level similarity
    combined_score = (similarity_weighted + sent_score) / 2

    # Normalize to [0,1] with clamping
    combined_score = torch.clamp((combined_score + 1) / 2, 0, 1).item()

    return combined_score
