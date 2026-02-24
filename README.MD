# 📊 Stakeholder Query Bot (RAG-Based)

A **Retrieval-Augmented Generation (RAG)** powered chatbot that answers stakeholder queries using internal project documents.

Built as a **Proof of Concept (POC)** using:

* **Streamlit** – for the UI
* **OpenAI LLM** – for natural language responses
* **ChromaDB** – vector database for document embeddings
* **LangChain** – for orchestrating the RAG pipeline

---

## 🚀 Problem Statement

Stakeholders often ask repetitive questions about:

* SLAs and escalation matrix
* Security policies
* Project timelines
* Technical architecture

Manually responding leads to:

* Delays
* Context switching
* Inconsistent answers

**Solution:** a chatbot that provides **document-grounded, instant, accurate answers**.

---

## 🧠 Architecture Overview

```
User Question
      ↓
Query Embedding
      ↓
Vector Search (ChromaDB)
      ↓
Top-K Relevant Document Chunks
      ↓
Context + LLM → Grounded Answer
      ↓
Answer Displayed in Streamlit
```

---

## 📂 Project Structure

```
stakeholder-query-bot/
│
├── app/
│   ├── main.py         # Streamlit UI
│   ├── ingest.py       # Document ingestion pipeline
│   ├── rag_pipeline.py # Retrieval + LLM logic
│   └── config.py       # Central configuration
│
├── data/raw/           # Dummy project documents
├── vectorstore/        # Auto-generated vector DB (gitignored)
│
├── .env                # API keys
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/stakeholder-query-bot.git
cd stakeholder-query-bot
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add OpenAI API Key

Create a `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

---

### 5️⃣ Add Documents

Place project documents inside:

```
data/raw/
```

Supported formats: `.pdf`, `.docx`, `.txt`

---

### 6️⃣ Run Ingestion

```bash
python app/ingest.py
```

This will create the **vector database**.

---

### 7️⃣ Start the Application

```bash
streamlit run app/main.py
```

---

## 💬 Example Questions

* What is the SLA for high priority issues?
* When is InsightX going live?
* What is the password policy?
* What database is used in the system?
* Who approves budget changes?

**Out-of-scope test:**

* What is the company’s annual revenue? → Should return:

> "I could not find this information in the knowledge base."

---

## 🔐 Guardrails

* Answers **strictly based on embedded documents**
* Out-of-context questions return a fallback response
* No hallucinations

---

## 🛠 Tech Stack

* Python
* Streamlit
* OpenAI (LLM + Embeddings)
* ChromaDB
* LangChain

---

## 🔮 Future Improvements

* Document upload via UI
* Highlighting source citations in answers
* Answer confidence scoring
* Deploy to Streamlit Cloud / Azure
* Add authentication
