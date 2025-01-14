from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
phi_api_key = os.getenv("PHI_API_KEY")

# Initialize Groq model with API key
groq_model = Groq(id="gpt-4o", api_key=groq_api_key)

# Initialize the Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions="Always include sources.",
    show_tool_calls=True,
    markdown=True,
)

# Initialize the Financial Agent
finance_agent = Agent(
    name="Financial AI Agent",
    model=groq_model,
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        ),
    ],
    instructions="Use tables to display the data.",
    show_tool_calls=True,
    markdown=True,
)

# Initialize the Multi-Agent
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources.", "Use tables to display the data."],
    show_tool_calls=True,
    markdown=True,
)

# Make a request
multi_ai_agent.print_response(
    "Summarize analyst recommendations and share the latest news for NVDA", stream=True
)
