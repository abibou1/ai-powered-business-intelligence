# core logic such as data processing, model training, and evaluation
# import necessary libraries for LangChain RAG

from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains import SequentialChain
from langchain.evaluation.qa import QAEvalChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain_experimental.agents import create_pandas_dataframe_agent

import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# analyse pre-prepared data and extract insights
df= pd.read_csv('data/sales_data.csv')

# Basic exploration
print(df.head()) # summary stats
print(df.info()) # Data types
# print("Print Columns:")
# print(df.columns.tolist())

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
sales_by_time = agent.invoke("What is sales performance by month? Group by Date month.")
product_analysis = agent.invoke("Top products by sales?")
regional_analysis = agent.invoke("Sales by region?")
customer_seg = agent.invoke("Segment customers by age or demographics if available.")
stats = agent.invoke("Calculate median, std dev of Sales.")

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

chain = prompt | llm

# Example run
context = retriever.invoke("sales trends")
response = chain.invoke({"input": "Identify key trends", "context": context})
print(response)

# Chain 1: Retrieve
retrieval_chain = PromptTemplate(template="Retrieve relevant data for: {query}") | llm

# Chain 2: Analyze
analysis_chain = PromptTemplate(template="Analyze: {data} for trends") | llm

overall_chain = (
    {"query": RunnablePassthrough()}
    | retrieval_chain
    | analysis_chain
)
result = overall_chain.invoke({"query": "sales patterns"})
print("Overall Analysis Result:", result)

# implement RAG System
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

response = qa_chain.invoke("What are the key business insights from the sales data?")
print("RAG Response:", response)

# Integrate memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Example conversation
response1 = conv_chain.invoke({"question": "Analyze sales trends"})
print("Conversation 1:", response1)
response2 = conv_chain.invoke({"question": "Based on that, recommend improvements"})
print("Conversation 2:", response2)

# Apply QAEvalChain to check model performance and accuracy
#from langchain.prompts import PromptTemplate
eval_chain = QAEvalChain.from_llm(llm)


# df['Date'].dt.month
examples = [
    {"query": "Total sales?", "answer": str(df['Sales'].sum())},
    {"query": "Sales by region?", "answer": str(df.groupby('Region')['Sales'].sum().to_dict())},
    # {"query": "Sales performance by month?", "answer": str(df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().to_dict())},
    {"query": "Sales performance by month?", "answer": str(df.groupby(df['Date'].dt.month)['Sales'].sum().to_dict())},
    {"query": "Calculate median, std dev of Sales?", "answer": f"Median: {df['Sales'].median()}, Std Dev: {df['Sales'].std()}"},
]

predictions = qa_chain.batch(examples)
graded = eval_chain.evaluate(examples, predictions)
print("Evaluation Results:", graded)

# Data visualization

# Sales trends over time
df.groupby(df['Date'].dt.month)['Sales'].sum().plot(kind='line')
plt.title('Sales Trends')
plt.savefig('sales_trends.png')

# Product comparisons (bar)
df.groupby('Product')['Sales'].sum().plot(kind='bar')
plt.savefig('product_performance.png')

# Regional (pie)
df.groupby('Region')['Sales'].sum().plot(kind='pie')
plt.savefig('regional_analysis.png')

# Customer demographics
df['Customer_Age'].hist()
plt.savefig('customer_demographics.png')

# __all__ = ["qa_chain", "agent", "memory", "conv_chain"]

