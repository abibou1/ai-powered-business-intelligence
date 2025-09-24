# core logic such as data processing, model training, and evaluation
# import necessary libraries for LangChain RAG

from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
# from langchain.chains import SequentialChain
from langchain.evaluation.qa import QAEvalChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

#from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain
from langchain.tools import Tool
from langchain_experimental.agents import create_pandas_dataframe_agent

import matplotlib.pyplot as plt

import pandas as pd


import os
from dotenv import load_dotenv


# load csv data
def load_data(csv_path: str = 'data/sales_data.csv') -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Clean up column names (fix spaces or hidden characters)
    df.columns = df.columns.str.strip()

    if df is None or df.empty:
        raise ValueError("CSV file failed to load or is empty")

    return df

# data exploration
def explore_data(df):
    print(df.head()) # summary stats
    print(df.info()) # Data types

def ensure_openai_env() -> str:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return openai_api_key


def compute_basic_metrics(df: pd.DataFrame) -> dict:
    metrics = {
        "total_sales": df['Sales'].sum(),
        "total_sales_by_region": df.groupby('Region')['Sales'].sum(),
    }
    return metrics

def build_documents(df: pd.DataFrame) -> list[Document]:
    # Convert each row of the pandas DataFrame (df) into a LangChain Document
    documents = [
        Document(
            page_content=(
                f"Date: {row['Date']}, Product: {row['Product']}, "
                f"Region: {row['Region']}, Sales: {row['Sales']}, "
                f"Customer_Age: {row['Customer_Age']}, "
                f"Customer_Gender: {row['Customer_Gender']}, "
                f"Customer_Satisfaction: {row['Customer_Satisfaction']}"
            ),
            metadata={"row_index": idx},
        )
        for idx, row in df.iterrows()
    ]

    # --- Add summary statistics document ---
    summary_stats = {
        "Total Sales": df['Sales'].sum(),
        "Mean Sales": df['Sales'].mean(),
        "Median Sales": df['Sales'].median(),
        "Std Dev Sales": df['Sales'].std(),
        "Min Sales": df['Sales'].min(),
        "Max Sales": df['Sales'].max(),
        "Sales by Region": df.groupby('Region')['Sales'].sum().to_dict(),
        "Sales by Product": df.groupby('Product')['Sales'].sum().to_dict(),
    }

    summary_doc = Document(
        page_content="Global Sales Statistics:\n"
        + "\n".join([f"{k}: {v}" for k, v in summary_stats.items()]),
        metadata={"type": "summary"},
    )

    documents.append(summary_doc)
    return documents


def build_retriever_from_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def get_llm(temperature: float = 0):
    # Low temp for factual answers
    return OpenAI(temperature=temperature)

def build_pandas_agent(llm, df: pd.DataFrame):
    return create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)


def run_example_agent_queries(agent):
    results = {
        "sales_by_time": agent.invoke("What is sales performance by month? Group by Date month."),
        "product_analysis": agent.invoke("Top products by sales?"),
        "regional_analysis": agent.invoke("Sales by region?"),
        "customer_seg": agent.invoke("Segment customers by age or demographics if available."),
        "stats": agent.invoke("Calculate median, std dev of Sales."),
    }
    return results

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

def build_prompt_chain(llm):
    prompt = PromptTemplate(
        input_variables=["input", "context"],
        template=(
            "Analyze this business data: {context}. Question: {input}. Provide insights and recommendations."
        ),
    )
    return prompt | llm


def run_prompt_chain(chain, retriever):
    context = retriever.invoke("sales trends")
    response = chain.invoke({"input": "Identify key trends", "context": context})
    return response

def build_overall_chain(llm):
    retrieval_chain = PromptTemplate(template="Retrieve relevant data for: {query}") | llm
    analysis_chain = PromptTemplate(template="Analyze: {data} for trends") | llm
    overall_chain = (
        {"query": RunnablePassthrough()}
        | retrieval_chain
        | analysis_chain
    )
    return overall_chain

def build_qa_chain(llm, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    return qa_chain

def build_conversational_chain(llm, retriever):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )
    return conv_chain

def evaluate_qa_chain(qa_chain, df: pd.DataFrame, llm):
    eval_chain = QAEvalChain.from_llm(llm)

    examples = [
        {"query": "Total sales?", "answer": str(df['Sales'].sum())},
        {"query": "Sales by region?", "answer": str(df.groupby('Region')['Sales'].sum().to_dict())},
        # {"query": "Sales performance by month?", "answer": str(df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().sort_index().to_dict())},
        {
            "query": "Calculate median, std dev of Sales?",
            "answer": f"Median: {df['Sales'].median()}, Std Dev: {df['Sales'].std()}",
        },
    ]

    predictions = qa_chain.batch(examples)
    graded = eval_chain.evaluate(examples, predictions)
    return graded

def generate_visualizations(df: pd.DataFrame):
    # Ensure Date is datetime
    df = df.copy()
    # df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # df.dropna(subset=['Date'], inplace=True)

    # Sales trends over time
    # plt.figure()
    # df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().plot(kind='line')
    # plt.title('Sales Trends')
    # plt.tight_layout()
    # plt.savefig('images/visualization_img/sales_trends.png')

    # Product comparisons (bar)
    plt.figure()
    df.groupby('Product')['Sales'].sum().plot(kind='bar')
    plt.tight_layout()
    plt.savefig('images/visualization_img/product_performance.png')

    # Regional (pie)
    plt.figure()
    df.groupby('Region')['Sales'].sum().plot(kind='pie')
    plt.tight_layout()
    plt.savefig('images/visualization_img/regional_analysis.png')

    # Customer demographics
    plt.figure()
    df['Customer_Age'].hist()
    plt.tight_layout()
    plt.savefig('images/visualization_img/customer_demographics.png')


# Initialize components for app.py to import
def initialize_components():
    """Initialize all components needed by app.py"""
    global qa_chain, agent, memory, conv_chain, df
    
    # Environment and data
    ensure_openai_env()
    df = load_data()
    
    # RAG components
    documents = build_documents(df)
    retriever = build_retriever_from_documents(documents)
    llm = get_llm(temperature=0)
    
    # Initialize components
    qa_chain = build_qa_chain(llm, retriever)
    agent = build_pandas_agent(llm, df)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conv_chain = build_conversational_chain(llm, retriever)
    
    return qa_chain, agent, memory, conv_chain, df


# Initialize components when module is imported
try:
    qa_chain, agent, memory, conv_chain, df = initialize_components()
except Exception as e:
    print(f"Warning: Failed to initialize components: {e}")
    # Set defaults to None to prevent import errors
    qa_chain = None
    agent = None
    memory = None
    conv_chain = None
    df = None


def main():
    # Use already initialized components or initialize if needed
    global qa_chain, agent, memory, conv_chain, df
    
    if df is None:
        qa_chain, agent, memory, conv_chain, df = initialize_components()
    
    explore_data(df)

    # Basic metrics
    metrics = compute_basic_metrics(df)
    print(f"Total Sales: {metrics['total_sales']}")
    print("Total Sales by Region:")
    print(metrics['total_sales_by_region'])

    # Agent queries (optional showcase)
    try:
        agent_results = run_example_agent_queries(agent)
        print(agent_results)
    except Exception as e:
        print(f"Agent queries failed: {e}")

    # RAG QA chain demo
    try:
        response = qa_chain.invoke("What are the key business insights from the sales data?")
        print("RAG Response:", response)
    except Exception as e:
        print(f"QA chain failed: {e}")

    # Conversational chain demo
    try:
        response1 = conv_chain.invoke({"question": "Analyze sales trends"})
        print("Conversation 1:", response1)
        response2 = conv_chain.invoke({"question": "Based on that, recommend improvements"})
        print("Conversation 2:", response2)
    except Exception as e:
        print(f"Conversational chain failed: {e}")

    # Evaluation
    try:
        llm = get_llm(temperature=0)
        graded = evaluate_qa_chain(qa_chain, df, llm)
        print("Evaluation Results:", graded)
    except Exception as e:
        print(f"Evaluation failed: {e}")

    # Visualizations
    try:
        generate_visualizations(df)
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    main()

