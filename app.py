import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.vectorstores import Milvus
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from pymilvus import connections
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# -------- CONFIG --------
connections.connect("default", host="localhost", port="19530")

app = FastAPI(title="Flow + RAG Chatbot")

# -------- MODELS --------
class ChatRequest(BaseModel):
    query: str
    mode: str  # "flow" or "rag"


# -------- COMPONENTS --------

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Initialize Milvus vector store for RAG
def init_vector_store(data_folder="sample_data"):
    all_docs = []

    #  Load all PDFs from folder
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(data_folder, file_name)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)

    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(all_docs)

    #  Create or connect Milvus collection
    vector_db = Milvus.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="rag_docs",
        connection_args={"host": "localhost", "port": "19530"},
    )

    print(f"✅ Loaded {len(splits)} chunks from {len(all_docs)} PDF documents.")
    return vector_db

vector_db = init_vector_store()
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
RAG_PROMPT = """
You are an advanced AI assistant designed to provide factual, well-structured answers grounded strictly in the provided context.

Follow these rules carefully:
1. Use ONLY the information from the context below.
2. If the answer cannot be found or inferred confidently, say: 
   "The provided context does not include that information."
3. Be concise, clear, and avoid repetition.
4. When helpful, summarize or reason over the context (don’t just quote).
5. Do not make up information or refer to yourself.

---

Context:
{context}

User Question:
{question}

---

Structured Response:
- **Answer:** <Your clear and factual answer>
- **Key Points:** <(optional) 2–3 bullet points summarizing reasoning or findings>
"""

prompt = PromptTemplate(
    template=RAG_PROMPT,
    input_variables=["context", "question"]
)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)


# -------- API ENDPOINT --------
@app.post("/chat")
async def chat(req: ChatRequest):
    query = req.query
    mode = req.mode.lower()

    if mode == "flow":
        # Simple guided flow chatbot (LLM only)
        prompt = f"You are a helpful assistant guiding the user. Answer this: {query}"
        response = llm.invoke(prompt)
        return {
            "mode": "flow",
            "response": response.content
        }

    elif mode == "rag":
        # Knowledge-based chatbot (RAG)
        result = rag_chain.invoke({"query": query})
        return {
            "mode": "rag",
            "response": result["result"],
            "sources": [doc.page_content for doc in result["source_documents"]],
        }

    else:
        return {"error": "Invalid mode. Use 'flow' or 'rag'."}



