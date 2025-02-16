# Recommendation Service using CatBoost & BERT Embeddings

## 📌 Overview
This project is an enhanced machine learning-based recommendation service that processes post text, extracts advanced features using BERT embeddings, and employs CatBoost to generate more accurate recommendations. The service is optimized for speed, integrating a database for efficient storage and exposing an API via FastAPI.

### 🔥 Updates in Version 2
✅ **BERT-based Feature Extraction** – Extracts semantic meaning from post text using `all-MiniLM-L6-v2` SentenceTransformer.  
✅ **Improved Accuracy** – Model accuracy increased by approximately **0.5%** with new features.  
✅ **Optimized Feature Engineering** – Added mean, max, and min of embeddings as new features.  
✅ **Efficient Storage** – Features stored in a database for optimized speed.  
✅ **API Service** – Built with FastAPI for seamless integration.  
✅ **Postman Integration** – Easily test API endpoints.  

---

## 📂 Project Structure



## 🚀 Feature Engineering
### **1. BERT-based Vectorization of Posts & Feature Extraction**
We leverage **BERT embeddings** to capture semantic similarities in post text.

#### **What Are Embeddings?**
Embeddings are numerical representations of text that capture the semantic meaning of words and phrases. Unlike traditional methods that rely on simple word frequency, embeddings encode contextual relationships between words. This helps in:
- Identifying similar posts even when different words are used.
- Understanding context beyond keyword matching.
- Improving recommendation accuracy by capturing deeper semantic patterns in text data.

---
