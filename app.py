from flask import Flask, request, render_template_string
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os
from langchain.prompts import PromptTemplate
# -----------------------------
# Step 1: Load & Prepare Docs (Only first time)
# -----------------------------
DB_PATH = "faiss_index"
DOC_PATH = "project rag.docx"

if not os.path.exists(DB_PATH):
    print("⚡ Building new FAISS index...")
    loader = Docx2txtLoader(DOC_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(DB_PATH)
else:
    print("✅ Loading FAISS index from disk...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# -----------------------------
# Step 2: Load Phi-2 Model (Quantized for speed)
# -----------------------------
model_name = "model/phi2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"   # quantization with bitsandbytes (fast!)
)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
)

local_llm = HuggingFacePipeline(pipeline=llm_pipeline)
QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant. 
Use the following context to answer the question. 
If the answer is not in the context, just say "I don't know". 
Do not repeat the context. Only output the final answer.

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"],
)

# -----------------------------
# Step 3: QA Chain (Optimized)
# -----------------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(
    llm=local_llm,
    retriever=retriever,
    chain_type="stuff" ,
     chain_type_kwargs={
        "prompt": QA_PROMPT,
        "document_variable_name": "context",   # ensures docs map correctly
    }, # faster & cleaner answers
)

# -----------------------------
# Step 4: Flask App
# -----------------------------
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Chatbot - Project Knowledge Base</title>
    <style>
        body { font-family: Arial, sans-serif; background:#f4f4f9; margin:0; padding:20px; }
        .chat-box { width: 600px; margin: auto; background: white; padding: 20px; border-radius: 12px; box-shadow: 0px 0px 12px rgba(0,0,0,0.1); }
        h2 { text-align: center; }
        form { margin-top: 20px; display:flex; }
        input[type=text] { flex:1; padding: 10px; border: 1px solid #ccc; border-radius: 8px; }
        button { padding: 10px 20px; margin-left:10px; background: #007BFF; color: white; border: none; border-radius: 8px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .response { margin-top: 20px; padding: 15px; background: #eef2ff; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="chat-box">
        <h2>🤖 Project Knowledge Base Chatbot</h2>
        <form method="POST">
            <input type="text" name="query" placeholder="Ask me something about the project..." required>
            <button type="submit">Ask</button>
        </form>
        {% if query %}
            <div class="response">
                <strong>You:</strong> {{ query }} <br><br>
                <strong>Bot:</strong> {{ response }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    query, response = None, None
    if request.method == "POST":
        query = request.form["query"]
        response = qa_chain.run(query)
    return render_template_string(HTML_TEMPLATE, query=query, response=response)

if __name__ == "__main__":
    app.run(debug=True)
