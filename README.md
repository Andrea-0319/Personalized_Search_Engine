# Personalized QA Retrieval & Recommendation System 🤖📚

## Aim of the project 🎯  
Build and evaluate a personalized retrieval pipeline over a large-scale community Q&A corpus (SE-PQA), demonstrating how lexical, semantic and user-specific signals can be combined to surface more relevant answers for each user.

**Note**: Check the Course project Proposal to better understand the full task.

## Description 📝  
This project uses the SE-PQA dataset (~1.1 M questions, 2.17 M answers, 589 k users across 50 StackExchange sites) to design and implement an end-to-end information retrieval and personalization workflow. From indexing with BM25 through neural re-ranking, query expansion, and tag-based personalization, we explore how each component contributes to ranking quality and user satisfaction.  

### Pipeline Overview:
1. **Data Preparation & Indexing** 🔧  
   - Clean and normalize text (case-folding, lemmatization)  
   - Build a unified PyTerrier index with Elasticsearch (BM25) first stage  
2. **Baseline & Parameter Tuning** ⚙️  
   - Grid search on BM25 parameters (k₁, b) to optimize Recall@100, P@1, MAP, NDCG  
3. **Neural Re-ranking** 🧠  
   - Embed and re-rank with MiniLM (cosine similarity)  
   - Fine-tune DistilBERT QA & MonoT5 for answer selection  
4. **Query Expansion** ✍️  
   - Generate expanded queries using T5-small (controlled max_new_tokens)  
5. **Personalization** 🏷️  
   - Compute tag-overlap scores between asker and answerer (“TAG” model)  (incomplete) 
6. **Fusion Strategies** 🔄  
   - Combine BM25, neural scores, expansion and TAG via Reciprocal Rank Fusion (RRF) or weighted sum  
7. **Evaluation** 📊  
   - Compare metrics across setups: P@1, Recall@100, MAP@100, NDCG@3  

### Objectives 
- Establish a strong BM25 baseline and tune its hyperparameters  
- Assess the impact of neural re-ranking models (MiniLM, DistilBERT QA, MonoT5)  
- Quantify gains from T5-based query expansion  
- Integrate lightweight tag-based personalization for user-aware ranking (incomplete)
- Devise effective fusion strategies to blend all signals  

### Main Results   
<img width="507" height="280" alt="Screenshot 2025-07-22 175144" src="https://github.com/user-attachments/assets/7f491d0e-6dfd-4def-a687-35997e45da3f" />

## Technologies Used 💻  
- **Python**  
- **PyTerrier** (retrieval pipeline)  
- **Elasticsearch / BM25** (first-stage indexing)  
- **HuggingFace Transformers** (MiniLM, DistilBERT QA, MonoT5, T5-small)  
- **Adapters** (parameter-efficient fine-tuning)  
- **ranx** (evaluation metrics)  
- **Jupyter Notebook** (experimentation & reporting)  
- Utility libraries: **pandas**, **NumPy**, **scikit-learn**, **json**, **pickle**
