import os
import streamlit as st
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Load API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Page configuration
st.set_page_config(page_title="Finance Agent", page_icon="📊", layout="wide")

# Custom CSS styling
st.markdown(
    """
    <style>
        .main { background-color: #f9f9f9; }
        .title { font-size: 40px; font-weight: bold; margin-bottom: 20px; }
        .footer { font-size: 14px; color: #999; text-align: center; margin-top: 40px; }
        .stTextInput > label { font-size: 16px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown("<div class='title'>💼 Financial Insights AI</div>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("🔍 Query Settings")
query = st.sidebar.text_input("Enter your financial query", "Latest news and stock details about Apple")
selected_model = st.sidebar.selectbox("Choose model", ["llama3-8b-8192", "llama3-70b-8192"])
run_query = st.sidebar.button("Run")

# Initialize the agent
agent = Agent(
    name="Finance AI Agent",
    role="Provides financial insights and web research",
    model=Groq(id=selected_model),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        ),
        DuckDuckGo()
    ],
    instructions=[
        "Use tables to display the data.",
        "Always include sources for news or financial data."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Execute the query
if run_query:
    with st.spinner("Analyzing financial data..."):
        try:
            response = agent.run(query)
            st.markdown(response.content)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown(
    "<div class='footer'>🔗 Powered by Streamlit, Phi, and Groq</div>",
    unsafe_allow_html=True
)
