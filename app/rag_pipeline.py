from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import re
try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate

from config import *

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

SMALL_TALK_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|how\s+are\s+you|thanks?|thank\s+you)\b",
    re.IGNORECASE,
)

PROMPT_TEMPLATE = """
You are an internal stakeholder support assistant.

Answer the question using ONLY the context below.
If not found, say:
"I could not find this information in the knowledge base."

Context:
{context}

Question:
{question}
"""


def _is_small_talk(query: str) -> bool:
    return bool(SMALL_TALK_PATTERN.search(query))


def _format_history(chat_history: list) -> str:
    if not chat_history:
        return ""

    recent = chat_history[-8:]
    history_lines = []
    for message in recent:
        role = message.get("role", "user")
        content = message.get("content", "")
        if not content:
            continue
        history_lines.append(f"{role.title()}: {content}")

    return "\n".join(history_lines)


def _build_standalone_query(query: str, chat_history: list) -> str:
    history_text = _format_history(chat_history)
    if not history_text:
        return query

    rewrite_prompt = ChatPromptTemplate.from_template(
        """
Rewrite the follow-up user question into a standalone search query for document retrieval.
Keep it concise and faithful to user intent.
If the question is already standalone, return it unchanged.

Conversation:
{history}

User question:
{question}
"""
    )

    rewrite_chain = rewrite_prompt | llm
    rewritten = rewrite_chain.invoke({"history": history_text, "question": query}).content.strip()

    return rewritten or query


def ask_question(query: str, chat_history: list | None = None):
    chat_history = chat_history or []

    if _is_small_talk(query):
        return "Hi! I’m doing well. Ask me any question about your uploaded documents, and I’ll answer with sources.", []

    retrieval_query = _build_standalone_query(query, chat_history)

    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(retrieval_query)
    else:
        docs = retriever.get_relevant_documents(retrieval_query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": query
    })

    sources = [doc.metadata.get("source", "Unknown") for doc in docs]

    return response.content, list(set(sources))
