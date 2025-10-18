# rag_api.py
from fastapi import FastAPI, Form, Request
from pydantic import BaseModel
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import uvicorn

# ---------------------------
# STEP 1: Initialize RAG Bot
# ---------------------------

print("🔄 Loading documents and building FAISS store...")
docs = load_all_documents("data")

store = FaissVectorStore("faiss_store")
store.build_from_documents(docs)
store.save()

rag_search = RAGSearch()

# ---------------------------
# STEP 2: FastAPI App Setup
# ---------------------------

app = FastAPI(title="RAG WhatsApp Chatbot")

# (Optional) Direct RAG API — keep your /ask endpoint for API queries too
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/ask")
def ask_portal(req: QueryRequest):
    response = rag_search.search_and_summarize(req.query, top_k=req.top_k)
    return {"answer": response}

# --------------------------------
# STEP 3: WhatsApp Webhook Endpoint
# --------------------------------

@app.post("/whatsapp")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    print(f"📩 Incoming WhatsApp message from {From}: {Body}")

    # Generate AI response
    try:
        response_text = rag_search.search_and_summarize(Body, top_k=2)
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        response_text = "⚠️ Sorry, something went wrong processing your request."

    # Create Twilio-compatible reply (TwiML)
    twilio_response = MessagingResponse()
    twilio_response.message(response_text)

    print(f"🤖 Replied to {From}: {response_text[:100]}...")

    # ✅ Return TwiML XML properly
    return Response(content=str(twilio_response), media_type="application/xml")


# ---------------------------
# STEP 4: Run the Server
# ---------------------------

if __name__ == "__main__":
    print("🚀 Starting WhatsApp RAG Chatbot on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


###.    uvicorn app:app --reload 

###.    ngrok http 8000 (in another tab) to expose local server for Twilio webhook URL