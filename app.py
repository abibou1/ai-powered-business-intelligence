
# Streamlit UI
import streamlit as st
import sys
import os
from src.assistant import qa_chain, agent, memory, conv_chain


st.title("InsightForge: AI Business Intelligence Assistant")

# Chat interface
user_input = st.text_input("Ask me anything about your business data:")
if user_input:
    response = conv_chain({"question": user_input})['answer'] # Use memory chain
    st.write(response)

# Visualization interface
st.subheader("Data Insights")
# st.image("images/visualization_img/sales_trends.png", caption="Sales Trends Over Time")
st.image("images/visualization_img/customer_demographics.png", caption="Customer Demographics")
st.image("images/visualization_img/product_performance.png", caption="Product Performance")
# st.image("images/visualization_img/regional_analysis.png", caption="Regional Analysis")

# Run agent for analysis
if st.button("Run Data Analysis Agent"):
    recs = agent.run("Provide business recommendations based on the data.")
    st.write(recs)

