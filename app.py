import os
import streamlit as st
from langchain_core.messages import HumanMessage

from agent import get_agent, content_to_text, guard_model, guard_prompt
import uuid

st.set_page_config(page_title="Food Nutrition Agent", page_icon="🥗", layout="centered")
st.title("Food Nutrition Agent")
st.write("Ask questions about food nutrition, dietary advice, and food labels.")


if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
elif not os.environ.get("GOOGLE_API_KEY"):
    st.error("Google API key is not configured.")
    st.info("Please enter your Google API key in the Streamlit app settings under 'Secrets'.")
    st.stop()

if "LANGSMITH_API_KEY" in st.secrets:
    os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
    os.environ["LANGSMITH_TRACING"] = "true"

@st.cache_resource
def load_cached_agent():
    return get_agent()

agent = load_cached_agent()
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())
thread_config = {"configurable": {"thread_id": st.session_state["thread_id"]}}


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if question := st.chat_input("Ask your question about food nutrition, dietary advice, or food labels..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("The agent is analyzing data and documents..."):
            try:
                result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=thread_config)
                answer = content_to_text(result["messages"][-1].content)

                guard_decision = guard_model.invoke([ HumanMessage(content=guard_prompt.format(agent_response=answer))])
                
                if guard_decision.verdict != "SAFE":
                    answer = ("I cannot provide personalized medical advice. Please consult a qualified healthcare professional.")

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:
                error_msg = f"An error occurred: {exc}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})