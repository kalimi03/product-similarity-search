import numpy as np
import pandas as pd
import faiss
import torch
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATA_PATH = "data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
FAISS_K = 50
WEIGHTS = {
    "text"    : 0.35,  # product name
    "brand"   : 0.20,
    "category": 0.20,  # parent category
    "numeric" : 0.15,  # price + rating
    "binary"  : 0.10,  # prime / bestseller flags
}

def _get_top_category(value):
    if isinstance(value, dict) and len(value) > 0:
        return list(value.keys())[0]
    return "Unknown"

# Data loading
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    # striping non-numeric characters (e.g. "$").
    df["sales_price"] = df["sales_price"].astype(str)
    df["sales_price"] = df["sales_price"].str.replace(r"[^\d.]", "", regex=True)
    df["sales_price"] = df["sales_price"].replace("", np.nan).astype(float)
    # assuming 999999999 a value meaning represting weight not provided.
    df["weight_known"]   = (df["weight"] != 999999999).astype(int)
    df["rating"]         = pd.to_numeric(df["rating"], errors="coerce") # convert bad values to NaN.
    df["is_prime"]       = (df["amazon_prime__y_or_n"].str.upper() == "Y").astype(int)
    df["is_best_seller"] = (df["best_seller_tag__y_or_n"].str.upper() == "Y").astype(int)
    df["top_category"]   = df["parent___child_category__all"].apply(_get_top_category)
    df["brand"]          = df["brand"].fillna("Unknown").astype(str)
    df["product_name"]   = df["product_name"].fillna("").astype(str)
    return df.reset_index(drop=True)

# Structured feature matrices
def _encode_categorical(df: pd.DataFrame, col: str) -> np.ndarray:
    labels = LabelEncoder().fit_transform(df[col])
    labels = labels.reshape(-1, 1)                   # shape required by MinMaxScaler
    return MinMaxScaler().fit_transform(labels)      # scale integers to [0, 1]

def _build_structured(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    # Fill missing prices/ratings with the column median before scaling
    num = df[["sales_price", "rating"]].copy()
    num["sales_price"] = num["sales_price"].fillna(num["sales_price"].median())
    num["rating"]      = num["rating"].fillna(num["rating"].median())
    return {
        "brand"   : _encode_categorical(df, "brand"),
        "category": _encode_categorical(df, "top_category"),
        "numeric" : MinMaxScaler().fit_transform(num),
        "binary"  : df[["weight_known", "is_prime", "is_best_seller"]].values.astype(float),
    }

# cosine similarity function
def calculate_similarity(
    query_idx: int,
    text_matrix: np.ndarray,
    structured: Dict[str, np.ndarray],
    weights: Dict[str, float],
    candidate_indices: np.ndarray,
) -> np.ndarray:
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6, "Weights must sum to 1.0"
    q_text = text_matrix[query_idx : query_idx + 1]
    c_text = text_matrix[candidate_indices]
    scores = weights["text"] * cosine_similarity(q_text, c_text).flatten()
    for name, matrix in structured.items():
        q_vec = matrix[query_idx : query_idx + 1]
        c_vec = matrix[candidate_indices]
        scores += weights[name] * cosine_similarity(q_vec, c_vec).flatten()
    return scores

# Top-n helper function
def top_n(scores, query_idx, n, uniq_ids, df):
    # Exclude the query product.
    scores[query_idx] = -1.0
    # table with score & other features for sorting.
    result = pd.DataFrame()
    result["uniq_id"]     = uniq_ids
    result["score"]       = scores
    result["rating"]      = df["rating"].fillna(0).values       # assuming missing 0 (worst)
    result["sales_price"] = df["sales_price"].fillna(9999).values  # assuming missing very expensive
    result = result.sort_values(
        by=["score", "rating", "sales_price"],
        ascending=[False, False, True]
    )
    top_results = result.head(n)
    return top_results["uniq_id"].tolist()

def build_faiss_index(embeddings: np.ndarray):
    """
    Build an IVFFlat FAISS index for approximate nearest neighbor search.
    """
    N, D = embeddings.shape
    nlist  = max(1, int(np.sqrt(N)))  # number of clusters to divide the vector space.
    nprobe = max(1, nlist // 4)       # cells to search at query time
    quantizer = faiss.IndexFlatIP(D)  # flat index using inner product
    index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    normed = embeddings.copy().astype(np.float32)
    faiss.normalize_L2(normed)
    index.train(normed)
    index.add(normed)
    index.nprobe = nprobe
    return index, normed

class BGEReranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3.
    """
    MODEL_NAME = "BAAI/bge-reranker-v2-m3"
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, documents: List[str]) -> np.ndarray:
        pairs  = [[query, doc] for doc in documents]
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            scores = self.model(**inputs).logits.view(-1).float().cpu().numpy()
        return scores

# Sentence-Transformer + FAISS + BGE Reranker
class SentenceTransformerIndex:
    MODEL = "all-MiniLM-L6-v2"
    def __init__(self, df: pd.DataFrame):
        self.df         = df
        self.uniq_ids   = df["uniq_id"].tolist()
        self.id_to_idx  = {uid: i for i, uid in enumerate(self.uniq_ids)}
        self.structured = _build_structured(df)
        self.text_matrix = SentenceTransformer(self.MODEL).encode(
            df["product_name"].tolist(),
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        # FAISS index on text embeddings
        self.faiss_index, self.normed_embeddings = build_faiss_index(self.text_matrix)
        # BGE reranker
        self.reranker = BGEReranker()

    def find_similar_products(self, product_id: str, num_similar: int) -> List[str]:
        idx        = self.id_to_idx[product_id]
        query_name = self.df["product_name"].iloc[idx]
        # Stage 1: FAISS approximate nearest neighbor
        query_vec = self.normed_embeddings[idx : idx + 1].copy()
        faiss.normalize_L2(query_vec)
        k = min(FAISS_K + 1, len(self.uniq_ids))
        _, faiss_indices = self.faiss_index.search(query_vec, k)
        candidates = faiss_indices[0]
        candidates = candidates[candidates != idx][:FAISS_K]  # exclude self
        # Stage 2: weighted structured similarity on candidates
        struct_scores = calculate_similarity(
            idx, self.text_matrix, self.structured, WEIGHTS, candidates
        )
        sorted_order = np.argsort(struct_scores)[::-1]
        candidates   = candidates[sorted_order]
        # Stage 3: BGE cross-encoder reranker
        candidate_names = self.df["product_name"].iloc[candidates].tolist()
        rerank_scores   = self.reranker.score(query_name, candidate_names)
        final_order     = np.argsort(rerank_scores)[::-1][:num_similar]
        top_indices     = candidates[final_order]
        return [self.uniq_ids[i] for i in top_indices]
