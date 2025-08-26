RAG-based Project Info Chatbot:

This project implements a Retrieval-Augmented Generation (RAG) chatbot that provides  insights about current project to developer rather than :

Talking to project manager everytime:

Short and concise answer related to modules deadline objectives etc.

The chatbot retrieves contextual knowledge from documents (e.g., research papers, financial reports, or stock guides) and combines it with LLM-powered reasoning for more accurate answers.

 Features:

Two LLM backends available:

app.py → Uses Microsoft Phi-2 (local model) for free offline usage.

app2.py → Uses Google Gemini API for cloud-based intelligent reasoning.

Document ingestion & vector search using FAISS.

Supports .txt and .docx data files as custom knowledge sources.

Streamlit Web UI for easy interaction.

RAG pipeline:

Load documents

Split into chunks

Embed & store in FAISS

Retrieve relevant context

Query the LLM with context + user question