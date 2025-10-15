import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant"
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, user_query: str, top_k: int = 5) -> str:
        # Retrieve relevant context from the vectorstore
        results = self.vectorstore.query(user_query, top_k=top_k)
        guide_context = "\n\n".join(
            [r["metadata"].get("text", "") for r in results if r["metadata"]]
        )

        if not guide_context:
            return "No relevant information found in the portal guide."

        # 🔥 Enhanced structured prompt
        prompt = f"""
You are a **Student Management Portal assistant** helping teachers and students.
Answer queries strictly based on the following guide context.

---
📘 Portal Guide Context:
{guide_context}

---
### 💬 User Query:
{user_query}

---
### 🧠 Instructions:
- Answer clearly and politely.
- If the question is related to portal usage, explain the exact steps.
- If it’s about academic or administrative policy, summarize the rule.
- If information is not in the guide, respond: *"This detail is not mentioned in the portal guide."*
- Avoid assumptions — stay factual.
- If your response includes numbered points (like 1., 2., 3.), place each point on a **new line** for clarity.

---



**Answer:**
- ...



"""

        # Generate answer from Groq LLM
        response = self.llm.invoke([prompt])
        return response.content.strip()


# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "How can I check my exam timetable on the portal?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
