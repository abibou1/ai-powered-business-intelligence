
# Streamlit UI
import streamlit as st
import sys
import os
from src.assistant import qa_chain, agent, memory, conv_chain


st.title("InsightForge: AI Business Intelligence Assistant")

# Check if components are initialized
if conv_chain is None or agent is None:
    st.error("Failed to initialize AI components. Please check your environment and data files.")
    st.stop()

# Chat interface
user_input = st.text_input("Ask me anything about your business data: ")
if user_input:
    try:
        response = conv_chain.invoke({"question": user_input})
        st.write(response['answer'])
    except Exception as e:
        st.error(f"Error processing your question: {e}")

# Visualization interface
st.subheader("Data Insights")
st.image("images/visualization_img/sales_trends.png", caption="Sales Trends Over Time")
st.image("images/visualization_img/customer_demographics.png", caption="Customer Demographics")
st.image("images/visualization_img/product_performance.png", caption="Product Performance")
st.image("images/visualization_img/regional_analysis.png", caption="Regional Analysis")

# Run agent for analysis
if st.button("Run Data Analysis Agent"):
    try:
        recs = agent.invoke("Provide business recommendations based on the data.")
        st.write(recs)
    except Exception as e:
        st.error(f"Error running analysis: {e}")

