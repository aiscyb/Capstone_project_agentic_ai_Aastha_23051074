import os
import uuid

import streamlit as st

from agent import DOCUMENTS, build_agent


st.set_page_config(page_title="E-Commerce Support Agent", page_icon="🛍️", layout="centered")
st.title("🛍️ E-Commerce Customer Support Assistant")
st.caption("Agentic RAG assistant for returns, refunds, shipping, and order policy help.")
DOMAIN_DESCRIPTION = (
    "Handles customer support queries for returns, refunds, shipping, tracking, cancellations, "
    "damaged items, and escalation guidance."
)

if "GROQ_API_KEY" in st.secrets and not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


@st.cache_resource
def load_agent():
    return build_agent(enforce_retrieval_gate=True, verbose=False)


try:
    assistant = load_agent()
except Exception as error:
    st.error(f"Failed to initialize agent: {error}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

with st.sidebar:
    st.subheader("Session")
    st.write(DOMAIN_DESCRIPTION)
    st.write(f"Thread ID: `{st.session_state.thread_id}`")
    st.write(f"KB Documents: `{assistant.collection.count()}`")
    st.write(f"Retrieval Score: `{assistant.retrieval_score:.2f}`")
    st.write(f"LLM Backend: `{assistant.llm_backend}`")
    st.markdown("**Topics covered**")
    for doc in DOCUMENTS:
        st.write(f"- {doc['topic']}")
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask about returns, refunds, shipping, or orders..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = assistant.ask(prompt, thread_id=st.session_state.thread_id)
            answer = result.get("answer", "I don't have that information in my knowledge base.")
            route = result.get("route", "unknown")
            faithfulness = result.get("faithfulness", 0.0)
            sources = result.get("sources", [])
        st.write(answer)
        st.caption(f"route={route} | faithfulness={faithfulness:.2f} | sources={sources}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
