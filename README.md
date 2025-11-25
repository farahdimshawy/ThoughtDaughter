# <h1>  Thought Daughter </h1>
<h2>Retrieval Augmented Generation (RAG) with Streamlit, LangChain and Pinecone</h2>
Hybrid RAG Chatbot: A Streamlit Q&amp;A app using Gemini, Pinecone, Wikipedia, and Tavily for real-time, hybrid retrieval. Combines internal document search with up-to-the-minute web search. Features transparent sourcing and an interactive chat interface.


<h2>Prerequisites</h2>
<ul>
  <li>Python 3.11+</li>
</ul>
<h2>Installation</h2>
1. Clone the repository:

```
git clone https://github.com/farahdimshawy/ThoughtDaughter.git
cd LangChain Pinecone RAG
```

2. Create a virtual environment

```
python -m venv venv
```

3. Activate the virtual environment

```
venv\Scripts\Activate
(or on Mac): source venv/bin/activate
```

4. Install libraries

```
pip install -r requirements.txt
```

5. Create accounts

- Create a free account on Pinecone: https://www.pinecone.io/
- Create an API key for OpenAI: https://platform.openai.com/api-keys
- Create an API key for Tavily API: https://app.tavily.com
  

6. **Set up your environment variables:**
   Create a `.env` file in the root of the project and add the following:
   ```
   PINECONE_API_KEY="your-pinecone-api-key"
   PINECONE_INDEX_NAME="your-pinecone-index-name"
   GOOGLE_API_KEY="your-google-api-key"
   TAVILY_API_KEY = "your-tavily-api-key"
   ```
<h3>Executing the scripts</h3>

1. Open a terminal in VS Code

2. Execute the following command:

```
python ingestion.py
python retrieval.py
streamlit run streamlit_app.py
```
