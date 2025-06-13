import streamlit as st
import logging
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Suppress verbose logs ---
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- Hardcoded Gemini API key ---
GEMINI_API_KEY = "AIzaSyCtD7pFRnyEX-0BxEvqI7QLpHl9fz_VWYw"

# --- Initialize Gemini chat model ---
def get_gemini_model():
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=GEMINI_API_KEY
        )
        return model
    except Exception as e:
        st.error(f"❌ Error initializing Gemini model: {e}")
        return None

# --- Initialize DuckDuckGo Search Tool ---
def get_search_tool():
    try:
        return DuckDuckGoSearchResults()
    except Exception as e:
        st.error(f"❌ Error initializing search tool: {e}")
        return None

# --- Initialize LangChain Agent ---
def init_agent(model, tools):
    try:
        agent = initialize_agent(
            tools=tools,
            llm=model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            max_iterations=3,
        )
        return agent
    except Exception as e:
        st.error(f"❌ Error initializing agent: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 Gemini Real-Time Q&A", page_icon="🔍", layout="centered")
st.title("🧠 Gemini Real-Time Q&A with 🔎 DuckDuckGo")
st.markdown("Ask anything about current events, trending news, or live facts! 🌍")

question = st.text_input("🔹 Type your question:", placeholder="e.g., Who won the T20 World Cup 2025?")

if st.button("Ask Gemini"):
    if not question.strip():
        st.warning("⚠️ Please enter a question before submitting.")
    else:
        with st.spinner("🔍 Searching and thinking..."):
            model = get_gemini_model()
            tool = get_search_tool()
            if model and tool:
                agent = init_agent(model, [tool])
                if agent:
                    try:
                        response = agent.run(question)
                        st.success("✅ Answer:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"❌ Failed to generate response: {e}")
