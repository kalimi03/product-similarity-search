import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
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
) -> np.ndarray:
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6, "Weights must sum to 1.0"
    scores = weights["text"] * cosine_similarity(
        text_matrix[query_idx:query_idx+1], 
        text_matrix).flatten()
    for name, matrix in structured.items():
        scores += weights[name] * cosine_similarity(
            matrix[query_idx:query_idx+1], 
            matrix).flatten()
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

# Sentence-Transformer
class SentenceTransformerIndex:
    MODEL = "all-MiniLM-L6-v2"
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.uniq_ids = df["uniq_id"].tolist()
        self.id_to_idx  = {uid: i for i, uid in enumerate(self.uniq_ids)}
        self.structured = _build_structured(df)
        self.text_matrix = SentenceTransformer(self.MODEL).encode(
            df["product_name"].tolist(),
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True)
    def find_similar_products(self, product_id: str, num_similar: int) -> List[str]:
        idx = self.id_to_idx[product_id]
        scores = calculate_similarity(idx, self.text_matrix, self.structured, WEIGHTS)
        return top_n(scores, idx, num_similar, self.uniq_ids, self.df)
