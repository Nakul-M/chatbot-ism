# rag_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import uvicorn
import os

# Load documents & initialize vectorstore
docs = load_all_documents("data")
store = FaissVectorStore("faiss_store")
store.build_from_documents(docs)
store.save()


rag_search = RAGSearch()

# Pydantic model for request body
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

app = FastAPI()

@app.post("/ask")
def ask_portal(req: QueryRequest):
    response = rag_search.search_and_summarize(req.query, top_k=req.top_k)
    return {"answer": response}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port) 
