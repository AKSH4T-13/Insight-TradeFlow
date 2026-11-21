import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from tools.data_analysis import (
    compare_groups,
    analyze_dataset,
    analyze_trends
)

load_dotenv()


def create_data_analyst():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        timeout=60,
        max_retries=0
    )

    tools = [
        compare_groups,
        analyze_dataset,
        analyze_trends
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
        You are an AI Data Analyst.

        Your job is to answer questions strictly using the provided CSV dataset.

        IMPORTANT:
        - Always use the provided file path when calling a tool.
        - Never invent a file path.
        - Never invent statistics or numerical results.
        - Use the appropriate tool to obtain actual data.
        - Explain findings clearly and concisely.
        - Revenue = Quantity * Price.
        - If a tool cannot answer the question, clearly explain the limitation.

        Available tools:

        1. get_dataset_summary
           Use for row count, columns, missing values and
           numerical statistics.

        2. get_category_revenue
           Use for category-wise revenue and highest/lowest
           revenue category.

        3. get_product_revenue
           Use for product-wise revenue and highest/lowest
           revenue product.

        4. get_city_revenue
           Use for city-wise revenue and highest/lowest
           revenue city.
        """
    )

    return agent
