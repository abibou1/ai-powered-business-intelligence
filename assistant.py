# core logic such as data processing, model training, and evaluation
# import necessary libraries for LangChain RAG
# from langchain import OpenAI
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain_experimental.agents import create_pandas_dataframe_agent

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# 1. analyse pre-prepared data and extract insights
df= pd.read_csv('data/sales_data.csv')

# Basic exploration
print(df.head()) # summary stats
print(df.info()) # Data types

# Calculate total sales
total_sales = df['Sales'].sum()
print(f"Total Sales: {total_sales}")

# Group by region and calculate total sales per region
total_sales_by_region = df.groupby('Region')['Sales'].sum()
print("Total Sales by Region:")
print(total_sales_by_region)

# Load CSV as LangChain documents
loader = CSVLoader('data/sales_data.csv')
documents = loader.load()

# Explore and organize the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Embed for retrieval
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Create a retriever from the vectorstore
retriever = vectorstore.as_retriever()

# Low temp for factual answers
llm = OpenAI(temperature=0)

# Pandas agent for data analysis
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

# Example queries
sales_by_time = agent.run("What is sales performance by month? Group by Date month.")
product_analysis = agent.run("Top products by sales?")
regional_analysis = agent.run("Sales by region?")
customer_seg = agent.run("Segment customers by age or demographics if available.")
stats = agent.run("Calculate median, std dev of Sales.")

# Custom Retriever: Tool to extract stats
def custom_stats_extractor(query):
    if "mean" in query.lower():
        return df['Sales'].mean()
    elif "median" in query.lower():
        return df['Sales'].median()
    elif "std dev" in query.lower() or "standard deviation" in query.lower():
        return df['Sales'].std()
    else:
        return "Statistic not recognized."
    
tool = Tool(name="StatsExtractor", func=custom_stats_extractor, description="Extracts statistical measures from sales data."  )

# Prompt Engineering
prompt = PromptTemplate(
    input_variables=["input", "context"],
    template="Analyze this business data: {context}. Question: {input}. Provide insights and recommendations."
)

chain = LLMChain(llm=llm, prompt=prompt)

# Example run
context = retriever.get_relevant_documents("sales trends")
response = chain.run(input="Identify key trends", context=context)
print(response)

