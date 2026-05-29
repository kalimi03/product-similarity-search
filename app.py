import os
import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from product_similarity import load_data, SentenceTransformerIndex
from product_similarity_advance import load_data, SentenceTransformerIndex


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DATA_PATH = os.getenv(
    "DATA_PATH",
    "data/marketing_sample_for_amazon_com-amazon_fashion_products"
    "__20200201_20200430__30k_data.ldjson",
)

# App state
state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and build index before the server starts accepting requests."""
    logger.info("Loading dataset from %s …", DATA_PATH)
    df = load_data(DATA_PATH)
    logger.info("Building Sentence-Transformer similarity index for %d products …", len(df))
    state["index"] = SentenceTransformerIndex(df)
    state["df"] = df
    logger.info("Index ready — service is up.")
    yield
    state.clear()

app = FastAPI(
    title="Product Similarity API",
    description="Returns similar Amazon fashion products given a product ID.",
    version="1.0.0",
    lifespan=lifespan,
)

# Response models
class SimilarProductsResponse(BaseModel):
    product_id : str
    num_similar: int
    results : List[str]

class ProductDetail(BaseModel):
    uniq_id : str
    product_name: str
    brand : str
    top_category: str
    sales_price : float | None
    rating : float | None
    is_prime : int
    is_best_seller: int

class HealthResponse(BaseModel):
    status : str
    products_indexed: int

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """app status check."""
    index_ready = "index" in state and "df" in state
    return {
        "status" : "ok" if index_ready else "not ok",
        "products_indexed": len(state["df"]) if index_ready else 0,
    }

@app.get("/find_similar_products", response_model=SimilarProductsResponse, tags=["similarity"])
def find_similar_products(
    product_id: str,
    num_similar: int = 5,
):
    """Return the top n similar products given product_id."""
    if "index" not in state:
        raise HTTPException(status_code=503, detail="Index not ready yet — try again shortly.")
    index = state["index"]
    if product_id not in index.id_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"product_id '{product_id}' not found in the dataset.",
        )
    try:
        results = index.find_similar_products(product_id, num_similar)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
    return {"product_id": product_id, "num_similar": num_similar, "results": results}

@app.get("/product/{product_id}", response_model=ProductDetail, tags=["products"])
def get_product(product_id: str):
    """Fetch the stored attributes for a single product."""
    if "df" not in state:
        raise HTTPException(status_code=503, detail="Service not ready.")
    df   = state["df"]
    rows = df[df["uniq_id"] == product_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"product_id '{product_id}' not found.")
    r = rows.iloc[0]
    return {
        "uniq_id"       : r["uniq_id"],
        "product_name"  : r["product_name"],
        "brand"         : r["brand"],
        "top_category"  : r["top_category"],
        "sales_price"   : None if str(r["sales_price"]) == "nan" else float(r["sales_price"]),
        "rating"        : None if str(r["rating"])      == "nan" else float(r["rating"]),
        "is_prime"      : int(r["is_prime"]),
        "is_best_seller": int(r["is_best_seller"]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
