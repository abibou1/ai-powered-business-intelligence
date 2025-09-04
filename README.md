
# InsightForge: AI-Powered Business Intelligence Assistant

InsightForge is an interactive AI-powered business intelligence tool that leverages advanced language models and retrieval-augmented generation (RAG) to analyze your business data, extract insights, and provide actionable recommendations. The project features a Streamlit-based UI for seamless interaction and visualization.

## Features

- **Conversational AI Assistant:** Ask questions about your business data and receive intelligent, context-aware answers.
- **Automated Data Analysis:** Get business recommendations and insights using AI agents.
- **Data Visualizations:** View key charts such as sales trends, customer demographics, product performance, and regional analysis.
- **Retrieval-Augmented Generation:** Combines data retrieval and generative AI for accurate, data-driven responses.

## Technologies Used

- Python
- Streamlit
- LangChain (including langchain-community, langchain-experimental, langchain-openai)
- OpenAI API
- Pandas
- FAISS
- dotenv

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (set in a `.env` file)
- All dependencies listed in `requirements.txt`

### Installation

1. **Clone the repository:**
	```sh
	git clone https://github.com/yourusername/ai-powered-business-intelligence.git
	cd ai-powered-business-intelligence
	```

2. **Create and activate a virtual environment:**
	```sh
	python -m venv venv
	.\venv\Scripts\activate
	```

3. **Install dependencies:(for linux environment)**
	```sh
	 pip install -r requirements.txt --no-cache-dir
	 pip freeze | grep -v "win" > clean-requirements.txt # remove windows specific packages
	 mv clean-requirements.txt requirements.txt
	```

4. **Set up your `.env` file:**
	```
	OPENAI_API_KEY=your_openai_api_key
	```

	**Docker**
	```
	docker build -t test-app .
	docker run -p 8501:8080 -e OPENAI_API_KEY=your_actual_key_here test-app
	```

5. **Prepare your data:**
	- Place your business data CSV file in the `data` directory (e.g., `data/sales_data.csv`).
	- Place visualization images in the `visualization_img` directory.

### Running the App

```sh
streamlit run app.py
```

## Usage

- Enter business-related questions in the chat interface.
- View data insights and visualizations.
- Click "Run Data Analysis Agent" for AI-generated recommendations.

## File Structure

```
ai-powered-business-intelligence/
│
├── app.py                # Streamlit UI
├── assistant.py          # Core logic, data processing, AI agents
├── requirements.txt      # Python dependencies
├── data/                 # Business data CSV files
├── visualization_img/    # Visualization images
└── README.md             # Project documentation
```

## Security Notice

This project uses AI agents capable of executing Python code. Only run in a secure, trusted environment. See [LangChain security guidelines](https://python.langchain.com/docs/security/) for details.

---

**InsightForge**: Unlock actionable business insights with AI.
