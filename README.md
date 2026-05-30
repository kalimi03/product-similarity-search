# Product Similarity Search

Product similarity search microservice built in 2 ways a simple one and one with FastAPI, FAISS, and BGE Reranker, both deployed on Kubernetes.

## Project Structure

```
├── app.py                        # FastAPI microservice
├── product_similarity.py         # simple similarity engine
├── product_similarity_advance.py # Two-stage similarity engine (FAISS + BGE Reranker)
├── requirements.txt              # Python dependencies for simple approach
├── requirements_advance.txt      # Python dependencies for advance approach
├── Dockerfile                    # Docker image definition
├── kubernetes/
│   └── deployment.yaml           # Kubernetes Deployment + Service
│── result_screenshots            # sample result screen shots at different stages of devployment.
└── data/
    └── marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson
```

---

## How It Works

The advance similarity search uses a two-stage pipeline:

- **Stage 1 — FAISS IVFFlat** (Approximate Nearest Neighbor): Retrieves top-50 candidate products using dense sentence embeddings
- **Stage 2 — BGE Cross-Encoder Reranker**: Rescores and reorders the 50 candidates for precise relevance ranking

---

## Prerequisites

- Python 3.10+
- Docker Desktop
- kubectl
- The dataset file is placed inside the `data/` folder

---

## Option 1 — Run Locally with Python

**Step 1 — Install dependencies:**
```bash
pip install -r requirements.txt
Or for advance search, install the below requirements
pip install -r requirements_advance.txt
```

**Step 2 — Start the server:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Step 3 — Open in browser:**
```
http://localhost:8000/docs
```

---

## Option 2 — Run with Docker

### Build the Docker Image

```bash
docker build -t simple-similarity-search:latest .
or
docker build -t advance-similarity-search:latest .

Note: Dont forget to update requirement.txt file name in docker file.
```

### Run the Container

```bash
docker run -p 8000:8000 -v $(pwd)/data:/app/data simple-similarity-search:latest
```

### Open in browser:
```
http://localhost:8000/docs
```

---

## Option 3 — Deploy on Kubernetes

### Prerequisites
- Docker Desktop with Kubernetes enabled
- kubectl installed

### Enable Kubernetes in Docker Desktop
1. Open Docker Desktop
2. Go to **Settings → Kubernetes**
3. Check **Enable Kubernetes**
4. Click **Apply & Restart**
5. Wait for the green Kubernetes status at the bottom

### Verify Kubernetes is running:
```bash
kubectl get nodes
```

Expected output:
```
NAME             STATUS   ROLES           AGE   VERSION
docker-desktop   Ready    control-plane   2m    v1.34.x
```

### Deploy the application:
```bash
kubectl apply -f kubernetes/deployment.yaml
```

### Check deployment status:
```bash
kubectl get pods
kubectl get svc
```

Expected output:
```
NAME                                  READY   STATUS    RESTARTS
product-similarity-xxx                1/1     Running   0
```

### Access the service:
```
http://localhost:80/docs
```

### Remove the deployment:
```bash
kubectl delete -f kubernetes/deployment.yaml
```

---

## API Endpoints

### 1. Find Similar Products
```
GET /find_similar_products
```

**HTTP Error Codes:**
| Code | Meaning                              |
|------|--------------------------------------|
| 404  | product_id not found in dataset      |
| 503  | Index not ready yet (still loading)  |
| 500  | Internal server error                |

---

### 2. Get Product Details
```
GET /product/{product_id}
```

---

### 3. Health Check
```
GET /health
```
---

**Test using Swagger UI:**
1. Open `http://localhost:8000/docs`
2. Click on `GET /find_similar_products`
3. Click **Try it out**
4. Enter `product_id` and `num_similar`
5. Click **Execute**

---

## Research References

The two-stage pipeline is based on the following papers:

| Paper | Authors | Link |
|-------|---------|------|
| Dense Passage Retrieval | Karpukhin et al., 2020 | https://arxiv.org/abs/2004.04906 |
| Billion-scale similarity search with GPUs (FAISS) | Johnson et al., 2017 | https://arxiv.org/abs/1702.08734 |
| Passage Re-ranking with BERT | Nogueira & Cho, 2019 | https://arxiv.org/abs/1901.04085 |
| BGE M3-Embedding | Chen et al., 2024 | https://arxiv.org/abs/2402.03216 |
| Sentence-BERT | Reimers & Gurevych, 2019 | https://arxiv.org/abs/1908.10084 |

---
