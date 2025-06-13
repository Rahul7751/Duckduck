import streamlit as st
import logging
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

# Suppress verbose logs
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

# 🔑 Hardcoded Gemini API Key (not recommended for production)
GEMINI_API_KEY = "AIzaSyCtD7pFRnyEX-0BxEvqI7QLpHl9fz_VWYw"

# Initialize Gemini model
def get_gemini_model():
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_API_KEY
        )
        return model
    except Exception as e:
        st.error(f"❌ Gemini initialization failed: {e}")
        return None

# Initialize DuckDuckGo search tool
def get_search_tool():
    try:
        return DuckDuckGoSearchResults()
    except Exception as e:
        st.error(f"❌ DuckDuckGo tool failed to initialize: {e}")
        return None

# Create the agent
def create_agent(model, tools):
    try:
        return initialize_agent(
            tools=tools,
            llm=model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )
    except Exception as e:
        st.error(f"❌ Failed to create agent: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="🧠 Ask Gemini", page_icon="🔍")
st.title("🧠 Gemini Real-Time Q&A with 🔎 DuckDuckGo")
st.markdown("Ask anything about current events, news, or real-world facts using Gemini + DuckDuckGo Search.")

query = st.text_input("🔹 Ask a question:", placeholder="e.g. What's the latest news on AI regulation?")
if st.button("Ask Gemini"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        with st.spinner("🤖 Gemini is thinking..."):
            model = get_gemini_model()
            tool = get_search_tool()
            if model and tool:
                agent = create_agent(model, [tool])
                if agent:
                    try:
                        answer = agent.run(query)
                        st.success("✅ Gemini's Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"❌ Failed to get answer: {e}")
