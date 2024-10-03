## README

### **Project: RAG Question-Answering System - Langchain**

### Description
This project implements a Retrieval Augmented Generation (RAG) system capable of answering questions based on a corpus of PDF documents. The system utilizes a combination of natural language processing techniques and a vector database to retrieve relevant information and generate informative responses.

### Dependencies
* Python 3.x
* Required libraries:
  * langchain
  * sentence-transformers
  * chromadb
  * google.generativeai
  * pandas
  * numpy
  * IPython

### Installation
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
* **API Keys:** Place your Gemini and LangSmith API keys in the `API/gemini_key` and `API/langSmith_key` files, respectively.
* **Data Folder:** Set the `source_data_folder` variable in the code to the path of the folder containing your PDF documents.
* **Vector Store Path:** Set the `path_db` variable in the code to the desired path for the vector store.

### Usage
1. Run the Python script.
2. Enter a query when prompted.
3. The system will retrieve relevant documents from the corpus and generate a response based on the query and the retrieved information.

### Customization
* **Model:** You can experiment with different language models and embedding models to tailor the system to your specific needs.
* **Retrieval Parameters:** Adjust the retrieval parameters (e.g., `score_threshold`, `k`) to control the number and relevance of retrieved documents.
* **Ranking Model:** Customize the cross-encoder model used for document ranking.
* **Prompt Engineering:** Experiment with different prompt templates to influence the generated responses.

### Notes
* Ensure you have the necessary API keys and data configured correctly before running the system.
* The system's performance may vary depending on the quality of the data and the complexity of the queries.
* For large datasets, consider optimizing retrieval and ranking techniques to improve efficiency.
