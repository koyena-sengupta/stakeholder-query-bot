📊 Stakeholder Query Bot (RAG-Based)

A Retrieval-Augmented Generation (RAG) powered chatbot that answers stakeholder queries using internal project documents.

Built as a Proof of Concept (POC) using:

Streamlit (UI)

OpenAI LLM

ChromaDB (Vector Store)

LangChain (RAG orchestration)

🚀 Problem Statement

Stakeholders often ask repetitive questions about:

SLAs

Escalation matrix

Security policies

Project timelines

Technical architecture

Manually responding to these queries leads to:

Delays

Context switching

Inconsistent responses

This solution enables:

Instant document-grounded answers

Reduced manual effort

Controlled hallucination via RAG

🧠 Architecture Overview

User Question
↓
Query Embedding
↓
Vector Search (ChromaDB)
↓
Top-K Relevant Chunks Retrieved
↓
Context Injected into LLM
↓
Grounded Answer Returned

📂 Project Structure
stakeholder-query-bot/
│
├── app/
│   ├── main.py              # Streamlit UI
│   ├── ingest.py            # Document ingestion pipeline
│   ├── rag_pipeline.py      # Retrieval + LLM logic
│   └── config.py            # Central configuration
│
├── data/raw/                # Dummy internal documents
├── vectorstore/             # Auto-generated embeddings (gitignored)
│
├── .env                     # API keys (not committed)
├── requirements.txt
└── README.md

⚙️ Setup Instructions
1️⃣ Clone Repository
git clone <your-repo-url>
cd stakeholder-query-bot

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add OpenAI API Key

Create a .env file:

OPENAI_API_KEY=your_api_key_here

5️⃣ Add Documents

Place internal documents inside:

data/raw/


Supported formats:

PDF

DOCX

TXT

6️⃣ Run Ingestion
python app/ingest.py


This creates the vector database.

7️⃣ Start the Application
streamlit run app/main.py

💬 Example Questions

What is the SLA for high priority issues?

When is InsightX going live?

What is the password policy?

What database is used in the system?

Who approves budget changes?

Out-of-scope test:

What is the company’s annual revenue?

🔐 Guardrails

Answers strictly based on embedded documents

No hallucinated responses

Out-of-context questions return a fallback response

🛠 Tech Stack

Python

Streamlit

OpenAI (LLM + Embeddings)

ChromaDB

LangChain

📌 Key Features

✔ Document-grounded responses
✔ Modular RAG pipeline
✔ Persistent vector storage
✔ Clean UI for demo
✔ Easy to extend

🔮 Future Improvements

Add document upload via UI

Add source citation highlighting

Add answer confidence score

Deploy to Azure / Streamlit Cloud

Add authentication