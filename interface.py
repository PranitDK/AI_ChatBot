# interface.py
import streamlit as st
import requests

# FastAPI backend endpoint
API_URL = "http://localhost:8000/chat"

st.title("AI Chatbot")

# Choose chat mode
mode = st.radio("Choose Mode", ["Flow Mode", "RAG Mode"])

# Input user query
query = st.text_input("Ask your question:")

if st.button("Send") and query:
    # Map mode names to backend-friendly values
    mode_value = "flow" if mode == "Flow Mode" else "rag"

    # Send request to FastAPI
    payload = {"query": query, "mode": mode_value}
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        data = response.json()
        st.markdown(f"**🗣️ Response:** {data.get('response', 'No answer')}")
        if mode_value == "rag" and "sources" in data:
            with st.expander("📚 Sources"):
                for src in data["sources"]:
                    st.markdown(f"- {src}")
    else:
        st.error(f"Error {response.status_code}: {response.text}")
