import streamlit as st
import pandas as pd

from src.tools import set_dataframe
from src.agent import ask_agent


st.set_page_config(
    page_title="AML Investigation Agent",
    page_icon="🔎",
    layout="wide",
)


st.title("🔎 AML Investigation Agent")

st.sidebar.header("Upload Transactions")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
)


if uploaded_file is None:
    st.info("Upload a transaction CSV to start the investigation.")
    st.stop()


try:
    df = pd.read_csv(uploaded_file)
    set_dataframe(df)

except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()


st.sidebar.success(
    f"Loaded {len(df):,} transactions"
)


with st.sidebar.expander("Dataset Preview"):
    st.dataframe(df.head(10))


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask an AML investigation question..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing..."):

            try:
                answer = ask_agent(question)

            except Exception as e:
                answer = f"Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )