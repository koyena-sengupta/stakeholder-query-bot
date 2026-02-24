import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(page_title="Stakeholder Query Bot", layout="wide")

st.title("Stakeholder Chatbot Assistant")
st.caption("Ask questions from your internal documents and get source-backed answers.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I’m your Stakeholder Chatbot Assistant. Ask me anything about your uploaded policies, SOPs, BRDs, or operations docs.",
            "sources": []
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown("**Sources**")
            for source in message["sources"]:
                st.write(f"- {source}")

query = st.chat_input("Ask your question...")

if query:
    prior_messages = st.session_state.messages.copy()

    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "sources": []
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            answer, sources = ask_question(query, prior_messages)
        st.markdown(answer)
        if sources:
            st.markdown("**Sources**")
            for source in sources:
                st.write(f"- {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
