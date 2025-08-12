"""
agents_demo.py

A demonstration of using LangChain agents with LLMs:
1. ReAct Agent - Integrates with Brave Search API to answer general queries.
2. Pandas DataFrame Agent - Performs data analysis directly on a CSV dataset.

Requirements:
- pip install langchain langchain-community langchain-openai langchain-experimental pandas python-dotenv
- Brave Search API key (https://brave.com/search/api/)
- OpenAI-compatible LLM API key
- BMI dataset (docs/bmi.csv) with 'Gender' and 'BMI' columns
"""

import os
import pandas as pd
from dotenv import load_dotenv

from langchain_community.tools import BraveSearch
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# ==============================
# 1. Load Environment Variables
# ==============================
load_dotenv()
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

# ==============================
# 2. Initialize LLM
# ==============================
def initialize_llm():
    """Initialize the ChatOpenAI model with environment variables."""
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0
    )

llm = initialize_llm()

# ==============================
# 3. Setup Tools
# ==============================
search_tool = BraveSearch.from_api_key(
    api_key=SEARCH_API_KEY,
    search_kwargs={"max_results": 5}
)
tools = [search_tool]

# ==============================
# 4. ReAct Agent
# ==============================
react_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are a ReAct-style agent. You can use the following tools:
{tools}

Tool names: {tool_names}

When answering, follow this format exactly:

Thought: describe your reasoning
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: result of the action

Repeat the Thought/Action/Action Input/Observation steps as needed.

If you have the final answer, write:
Final Answer: your final answer

Question: {input}
{agent_scratchpad}
"""
)

def create_react_agent_executor(llm, tools, prompt):
    """Create an AgentExecutor for the ReAct Agent."""
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

react_agent = create_react_agent_executor(llm, tools, react_prompt)

# ==============================
# 5. Pandas DataFrame Agent
# ==============================
def create_pandas_agent(llm, df):
    """Create a Pandas DataFrame agent for in-LLM data analysis."""
    return create_pandas_dataframe_agent(
        llm=llm, 
        df=df, 
        verbose=True, 
        agent_type=AgentType.OPENAI_FUNCTIONS, 
        allow_dangerous_code=True
    )

df = pd.read_csv("docs/bmi.csv")
pandas_agent = create_pandas_agent(llm, df)

# ==============================
# 6. Demo Execution
# ==============================
if __name__ == "__main__":
    print("\n=== ReAct Agent Demo (Web Search) ===")
    react_response = react_agent.invoke({
        "input": "What are the health benefits of regular exercise?"
    })
    print("\nFinal Answer:", react_response["output"])

    print("\n=== Pandas Agent Demo (Data Analysis) ===")
    pandas_response = pandas_agent.invoke(
        "Plot a bar chart of the average BMI by gender."
    )
    print("\nAnalysis Output:", pandas_response)