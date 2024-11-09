# %%
# pip install googleapis-common-protos==1.56.2

# %%
# pip install protobuf==3.20.3

# %%
"""
## **RAG with Langchain - Why Use LangChain?**

LangChain, a framework designed for building applications powered by large language models (LLMs), offers several key advantages that are evident in the provided code:

#### 1. **Modular and Extensible Architecture:**

* **Component-Based Approach:** LangChain's modular design allows for easy integration of different components, such as document loaders, text splitters, vector stores, embeddings, and language models. This flexibility enables you to customize and tailor the system to your specific needs.
* **Customizable Prompts:** The framework provides tools to create and manage prompts, which are the instructions given to the LLM. This allows you to fine-tune the LLM's responses and guide its behavior.

#### 2. **Efficient Data Handling:**

* **Document Loading and Processing:** LangChain offers efficient methods for loading and processing documents, including PDF files. This simplifies the process of extracting relevant information from large datasets.
* **Text Splitting:** The `RecursiveCharacterTextSplitter` used in the code ensures that text is split into manageable chunks while preserving context. This is crucial for effective processing by LLMs.
* **Vector Database:** The use of a vector database (Chroma) enables efficient storage and retrieval of embeddings, which are numerical representations of text. This facilitates semantic search and similarity matching.

#### 3. **Integration with LLMs:**

* **Language Model Interaction:** LangChain provides seamless integration with various LLMs, including Google's Gemini. This allows you to leverage the power of these models for tasks like question answering, summarization, and text generation.
* **Prompt Engineering:** The framework's support for prompt engineering helps you craft effective prompts that guide the LLM's responses and improve the quality of the generated content.

#### 4. **Workflow Management:**

* **RAG Chains:** The ability to create RAG (Retrieval Augmented Generation) chains allows you to combine document retrieval, prompt engineering, and LLM generation into a cohesive workflow. This streamlines the process of building question-answering systems and other applications.

#### 5. **Extensibility and Customization:**

* **Custom Components:** LangChain allows you to create custom components, such as custom retrievers or output parsers, to tailor the framework to your specific requirements.
* **Integration with Other Tools:** You can easily integrate LangChain with other tools and libraries to enhance its capabilities.

**In summary, LangChain provides a robust and flexible framework for building applications powered by LLMs. Its modular architecture, efficient data handling, seamless integration with LLMs, and extensibility make it a valuable tool for a wide range of natural language processing tasks.**

"""

# %%
"""
### Import Libraries
"""

# %%
"""
**Purpose:**

This code imports necessary libraries for working with PDF documents, natural language processing, and vector databases. It specifically leverages LangChain, a framework for building applications powered by large language models.

**Key Libraries and Their Functions:**

* **Generic Libraries:**
  - `os`: Provides functions for interacting with the operating system, such as file and directory operations.
  - `IPython.display`: Enables displaying various types of output, including Markdown, in Jupyter Notebooks.

* **Data Preparation Libraries:**
  - `langchain.document_loaders.PyPDFDirectoryLoader`: Loads PDF documents from a directory.
  - `langchain.text_splitter.RecursiveCharacterTextSplitter`: Splits text into smaller chunks based on character count.
  - `langchain.vectorstores.Chroma`: Creates a vector database for storing and retrieving embeddings.
  - `langchain.embeddings.SentenceTransformerEmbeddings`: Computes embeddings for text using a SentenceTransformer model.

* **Data Retrieval Libraries:**
  - `langchain.hub`: Provides access to pre-trained language models and tools.
  - `langchain_core.output_parsers.StrOutputParser`: Parses output as strings.
  - `langchain_core.runnables.RunnablePassthrough`: Passes input through without modification.
  - `langchain_google_genai.ChatGoogleGenerativeAI`: Interacts with Google's generative AI model.
  - `langchain.prompts.ChatPromptTemplate`: Creates chat-based prompts for language models.

* **Additional Libraries:**
  - `google.generativeai`: Provides access to Google's generative AI APIs.
  - `sentence_transformers.CrossEncoder`: Computes similarity scores between pairs of sentences.

**Overall Functionality:**

This code appears to be setting up the environment for building an application that processes PDF documents, extracts information from them, and uses a large language model to generate responses or perform other tasks. The libraries provide the necessary tools for loading, splitting, embedding, and storing text data, as well as interacting with a generative AI model.


- **Pandas:** A powerful library for working with structured data, providing data structures like DataFrames and Series, along with various functions for data cleaning, manipulation, and analysis.
- **NumPy:** A fundamental library for numerical computations, offering efficient multi-dimensional arrays and matrices, as well as mathematical functions and operations.
- **JSON:** A library for working with JSON (JavaScript Object Notation) data, which is a popular format for data exchange.

**Key Features and Use Cases:**

**Pandas:**

- **Data Structures:**
  - DataFrames: Two-dimensional labeled data structures similar to spreadsheets.
  - Series: One-dimensional labeled arrays.
- **Data Manipulation:**
  - Filtering, sorting, grouping, and aggregating data.
  - Handling missing data and outliers.
  - Joining and merging dataframes.
- **Data Analysis:**
  - Statistical calculations and visualizations.
  - Time series analysis.

**NumPy:**

- **Arrays:**
  - Efficient storage and manipulation of numerical data in multi-dimensional arrays.
  - Mathematical operations on arrays.
- **Linear Algebra:**
  - Matrix operations, solving linear equations, and eigenvalue problems.
- **Random Number Generation:**
  - Generating random numbers and arrays for simulations and statistical analysis.

**JSON:**

- **Parsing:**
  - Converting JSON strings into Python objects (dictionaries and lists).
- **Serialization:**
  - Converting Python objects into JSON strings.
- **Data Exchange:**
  - Interacting with APIs and other systems that use JSON.
"""

# %%
# Generic Libraries
import os
from IPython.display import display, Markdown

# Data Preparation Libraries
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Data Retrieval libraries
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

import google.generativeai as genai
from sentence_transformers import CrossEncoder

# %%
import pandas as pd
import numpy as np
import json

# %%
"""
This code sets the maximum column width in pandas DataFrames to `None`, which means that columns can display content of any length without truncation.
"""

# %%
pd.set_option('display.max_colwidth', None)

# %%
"""
### Load the keys
"""

# %%
"""
**Purpose:**

This code snippet loads API keys for Gemini and LangSmith, sets environment variables for LangChain tracing, and configures the API keys for use with LangChain.

**Explanation:**

- **`gemini_key = open("API/gemini_key", "r").read()`:** This line reads the contents of the file "API/gemini_key" and stores it in the `gemini_key` variable. This file is assumed to contain the API key for Gemini.
- **`langSmith_key = open("API/langSmith_key", "r").read()`:** This line reads the contents of the file "API/langSmith_key" and stores it in the `langSmith_key` variable. This file is assumed to contain the API key for LangSmith.
- **`os.environ["LANGCHAIN_TRACING_V2"] = "true"`:** This line sets the environment variable `LANGCHAIN_TRACING_V2` to "true". This enables tracing in LangChain, which can be helpful for debugging and monitoring the execution of your code.
- **`os.environ["LANGCHAIN_API_KEY"] = langSmith_key`:** This line sets the environment variable `LANGCHAIN_API_KEY` to the value of `langSmith_key`. This configures LangChain to use the LangSmith API.
- **`os.environ["GOOGLE_API_KEY"] = gemini_key`:** This line sets the environment variable `GOOGLE_API_KEY` to the value of `gemini_key`. This configures LangChain to use the Gemini API.

**Impact:**

After running this code, LangChain will be configured to use the specified API keys for Gemini and LangSmith. Tracing will also be enabled, allowing you to monitor the execution of your code and identify potential issues.
"""

# %%
gemini_key = open("API/gemini_key", "r").read()
langSmith_key = open("API/langSmith_key", "r").read()

# %%
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = langSmith_key
os.environ["GOOGLE_API_KEY"] = gemini_key

# %%
"""
### Load the document
"""

# %%
"""
**Purpose:**

This code snippet loads PDF documents from a specified directory and counts the total number of documents loaded.

**Explanation:**

- **`source_data_folder = "documents"`:** This line sets the variable `source_data_folder` to the string "documents". This indicates the directory where the PDF documents are located.
- **`loader = PyPDFDirectoryLoader(source_data_folder)`:** This line creates an instance of the `PyPDFDirectoryLoader` class, passing the `source_data_folder` as an argument. This class is responsible for loading PDF documents from the specified directory.
- **`data_on_pdf = loader.load()`:** This line calls the `load` method on the `loader` object. This method loads all the PDF documents from the directory and returns a list of `Document` objects. The list of documents is stored in the `data_on_pdf` variable.
- **`len(data_on_pdf)`:** This line calculates the length of the `data_on_pdf` list. This gives you the total number of PDF documents that were loaded from the directory.

**Impact:**

After running this code, the `data_on_pdf` variable will contain a list of `Document` objects, each representing a PDF document loaded from the specified directory. The `len(data_on_pdf)` expression will give you the count of these documents.

"""

# %%
source_data_folder = "documents"

loader = PyPDFDirectoryLoader(source_data_folder)
data_on_pdf = loader.load()

len(data_on_pdf)

# %%
"""
### Split the document and form using RecursiveCharacterTextSplitter
"""

# %%
"""
**Purpose:**

This code snippet partitions a list of PDF documents into smaller chunks of text, while preserving context by maintaining overlaps between chunks.

**Explanation:**

- **`text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=200)`:** This line creates an instance of the `RecursiveCharacterTextSplitter` class, configuring it with the following parameters:
  - `separators`: A list of characters or strings that will be used to split the text. In this case, the separators are newline characters, periods, spaces, and empty strings.
  - `chunk_size`: The desired size of each text chunk in characters. Here, it's set to 1000.
  - `chunk_overlap`: The number of characters that should overlap between adjacent chunks. Here, it's set to 200.
- **`splits = text_splitter.split_documents(data_on_pdf)`:** This line calls the `split_documents` method on the `text_splitter` object, passing the `data_on_pdf` list as an argument. This method splits each PDF document in the list into smaller chunks based on the specified parameters and returns a list of `Document` objects.
- **`len(splits)`:** This line calculates the length of the `splits` list. This gives you the total number of text chunks that were generated from the PDF documents.

**Impact:**

After running this code, the `splits` variable will contain a list of `Document` objects, each representing a text chunk extracted from one of the PDF documents. The length of `splits` will indicate the total number of chunks created.

"""

# %%
# Partitioning the data. With a limited size (chunks) 
# and 200 characters of overlapping to preserve the context
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(data_on_pdf)
# Number of Chunks generated
len(splits)

# %%
"""
### Create a new embedding model `all-MiniLM-L6-v2` and Create a VectorDB and store the documents split in it in the the embedded form
"""

# %%
"""
**Purpose:**

This code sets up the creation of text embeddings for the previously split text chunks (`splits`) and stores them in a vector database.

**Explanation:**

- **`embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")`:**
  - This line initializes an instance of the `SentenceTransformerEmbeddings` class.
  - This class is responsible for generating vector representations (embeddings) for text using pre-trained sentence transformer models.
  - The `model_name` parameter specifies the pre-trained model to be used. Here, it's set to "all-MiniLM-L6-v2", which is a Sentence Transformer model based on the MiniLM-L6-v2 language model from Hugging Face (link provided).

**Note:**

- You can choose different pre-trained models available on Hugging Face ([invalid URL removed]) depending on your specific needs and the domain of your text data.

- **`path_db = "langchain_store"`:** This line sets the variable `path_db` to the string "langchain_store". This specifies the directory where the database will be stored.

- **`vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model, persist_directory=path_db)`:**
  - This line creates an instance of the `Chroma` class, which is a vector database used by LangChain.
  - The `from_documents` method is used to initialize the database with the provided documents (`splits`) and the chosen embedding model (`embeddings_model`).
  - The `persist_directory` argument specifies the directory (`path_db`) where the database will be persisted on disk. This ensures that the database contents are saved and can be reused across sessions.

**Impact:**

After running this code, the `vectorstore` object will contain a vector database. Each document (text chunk) from the `splits` list will be stored as a key, along with its corresponding embedding vector generated by the `embeddings_model`. This allows you to efficiently retrieve documents based on their semantic similarity by querying the vector database.

**Benefits of Using a Vector Database:**

- Fast retrieval of similar documents based on their embeddings.
- Efficient storage and management of large collections of text data.
- Enables powerful information retrieval and search functionalities.
"""

# %%
# For the creation of the embeddings we will use Hugging Face
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# You can use any other model
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# %%
# Database folder path
path_db = "langchain_store" # @param {type:"string"}
#  Store the chunks in the DataBase
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model, persist_directory=path_db)

# %%
"""
### Create LLM and VectorStore retriever
"""

# %%
"""
**Purpose:**

This code snippet creates a `Retriever` object from a given `Chroma` vector store. The `Retriever` object provides an interface for retrieving documents from the vector store based on semantic similarity.

**Explanation:**

- **`vectorstore.as_retriever()`:** This method is called on the `Chroma` object `vectorstore`. It converts the vector store into a `Retriever` object. The `Retriever` object encapsulates the vector store and provides a convenient way to interact with it for document retrieval.

**Impact:**

After running this code, the `retriever` variable will contain a `Retriever` object. This object can be used to query the vector store for documents that are semantically similar to a given query.ch as specifying the number of documents to retrieve or adjusting the similarity threshold.

By using the `Retriever` object, you can easily query the vector store for relevant documents based on their semantic content.

"""

# %%
retriever = vectorstore.as_retriever()

# %%
"""
**Purpose:**

This code snippet creates an instance of the `ChatGoogleGenerativeAI` class, which represents a conversational language model based on Google's Gemini model. The configuration parameters are set to control the generation process.

**Explanation:**

- **`gen_config = genai.types.GenerationConfig(candidate_count=1)`:**
  - This line creates an instance of the `GenerationConfig` class from the `genai.types` module.
  - The `candidate_count` parameter is set to 1, which means that the model will only generate one text response for each prompt.

- **`llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None, max_retries=2, generation_config = gen_config, api_key=gemini_key)`:**
  - This line creates an instance of the `ChatGoogleGenerativeAI` class. The following parameters are specified:
    - `model`: The name of the language model to use. In this case, it's set to "gemini-1.5-pro", indicating the Gemini 1.5 Pro model.
    - `temperature`: The temperature parameter controls the randomness of the generated text. A temperature of 0 means that the model will always generate the most likely response.
    - `max_tokens`: The maximum number of tokens (words or subwords) that the model can generate in a single response. If set to `None`, there is no limit.
    - `timeout`: The maximum time (in seconds) that the model can take to generate a response. If set to `None`, there is no limit.
    - `max_retries`: The maximum number of times the model will retry generating a response if an error occurs.
    - `generation_config`: The `GenerationConfig` object created earlier, which specifies the desired number of candidates for each response.
    - `api_key`: The API key for accessing the Gemini model.

**Impact:**

After running this code, the `llm` variable will contain an instance of the `ChatGoogleGenerativeAI` class. This object can be used to interact with the Gemini model and generate text responses based on prompts. The configuration parameters will control the generation process, ensuring that the model generates only one response and uses the specified temperature and token limits.
"""

# %%
gen_config = genai.types.GenerationConfig(candidate_count=1)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                         temperature=0,
                         max_tokens=None,
                         timeout=None,
                         max_retries=2,
                         generation_config = gen_config,
                         api_key=gemini_key
                         )
llm

# %%
# https://smith.langchain.com/hub/rlm/rag-prompt
# prompt = hub.pull("rlm/rag-prompt")
# prompt

# %%
"""
### Business logic
"""

# %%
"""
**`get_store_results` function:**

- **Purpose:** Retrieves relevant documents from the vector store based on a given query.
- **Parameters:**
  - `question`: The query string to search for.
- **Returns:**
  - A tuple containing:
    - `docs`: A list of relevant documents retrieved from the vector store.
    - `question`: The original query string.
- **Implementation:**
  - Sets the search keyword arguments to specify the minimum score threshold and the maximum number of documents to retrieve.
  - Calls the `get_relevant_documents` method of the `retriever` object to retrieve documents based on the query and search keyword arguments.
  - Returns the retrieved documents and the original query.

**`cross_encoder_ranking` function:**

- **Purpose:** Reranks the retrieved documents using a cross-encoder model to improve relevance.
- **Parameters:**
  - `docs`: A list of retrieved documents.
  - `question`: The original query string.
- **Returns:**
  - A DataFrame containing the reranked documents with their metadata and page content.
- **Implementation:**
  - Loads a pre-trained cross-encoder model from Hugging Face.
  - Creates a DataFrame from the retrieved documents, extracting their metadata and page content.
  - Converts the metadata to strings for compatibility with the cross-encoder model.
  - Creates a list of input pairs for the cross-encoder, consisting of the query and each document's page content.
  - Predicts the relevance scores for each input pair using the cross-encoder model and stores them in the DataFrame.
  - Removes duplicate documents from the DataFrame.
  - Sorts the DataFrame by the reranked scores in descending order.
  - Returns the DataFrame containing the reranked documents.

**`results_runnable` function:**

- **Purpose:** Combines the `get_store_results` and `cross_encoder_ranking` functions to retrieve and rerank documents based on a given query.
- **Parameters:**
  - `question`: The query string to search for.
- **Returns:**
  - A DataFrame containing the reranked documents with their metadata and page content.
- **Implementation:**
  - Calls the `get_store_results` function to retrieve relevant documents.
  - Calls the `cross_encoder_ranking` function to rerank the retrieved documents.
  - Returns the reranked documents as a DataFrame.

**Overall Functionality:**

The `results_runnable` function provides a pipeline for retrieving and reranking documents based on a given query. It first retrieves relevant documents from the vector store using the `get_store_results` function and then reranks them using a cross-encoder model to improve relevance. The final result is a DataFrame containing the reranked documents with their metadata and page content.

"""

# %%
def get_store_results(question):
    search_kwargs = {"score_threshold":0.8,"k":10}
    docs = retriever.get_relevant_documents(query=question, search_kwargs=search_kwargs)
    return docs, question
    
def cross_encoder_ranking(docs, question):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    res_df = pd.DataFrame([t.__dict__ for t in docs])[["metadata", "page_content"]]
    res_df.metadata = res_df.metadata.apply(lambda x: str(x))
    cross_inputs = [[question, response.page_content] for response in docs]
    res_df["Reranks"] = cross_encoder.predict(cross_inputs)
    res_df = res_df.drop_duplicates()
    res_df = res_df.sort_values(by='Reranks', ascending=False)
    return res_df[["metadata", "page_content"]]
    

# %%
"""
### Pipeline to run search and reranking
The `results_runnable` function provides a pipeline for retrieving and reranking documents based on a given query. It first retrieves relevant documents from the vector store using the `get_store_results` function and then reranks them using a cross-encoder model to improve relevance. The final result is a DataFrame containing the reranked documents with their metadata and page content.
"""

# %%
def results_runnable(question):
    # question = "What can you tell me about life insurance premiums? "
    docs, question = get_store_results(question)
    result = cross_encoder_ranking(docs, question)

    return result

# %%
question = "What are the age related conditions in the life insurance?"
temp_df = results_runnable(question)

# %%
temp_df

# %%
"""
### Prompt for the LLM
- Input variables - `question` and `context`
"""

# %%
prompt = ChatPromptTemplate.from_messages(
    [("system", """
You are a highly skilled insurance expert tasked with answering user queries using the provided search results. These results are one or more pages from relevant insurance documents that contain the information needed to address the query.

You have a user query: '{question}'. The relevant search results are in the DataFrame '{context}'. The 'page_content' column contains the text from the policy documents, and the 'metadata' column contains the policy name and source page.
        **Your Task:**
        1. **Analyze the Query:** Carefully understand the user's intent and the specific information they are seeking.
        2. **Identify Relevant Documents:** Select the most pertinent documents from the search results based on their content and relevance to the query.
        3. **Extract Key Information:** Carefully extract the required information from the selected documents, ensuring accuracy and completeness.
        4. **Construct a Comprehensive Response:** Craft a clear, concise, and informative response that directly addresses the user's query.
        5. **Provide Citations:** Cite the specific policy names and page numbers where the information was found, using the following format:

            **[Policy Name], [Page Number]**

            **References:**
            * [Policy Name 1], [Page Number 1]
            * [Policy Name 2], [Page Number 2]
            * ...

        **Guidelines:**
        * **Accuracy:** Ensure that your response is factually correct and consistent with the information provided in the documents.
        * **Relevance:** Focus on the most relevant information and avoid providing unnecessary details.
        * **Clarity:** Use plain language and avoid technical jargon.
        * **Completeness:** Provide a comprehensive answer that covers all aspects of the user's query.
        * **Conciseness:** Be brief and to the point, while still providing sufficient detail.

        **Example Response:**
        > The maximum coverage for [policy type] is [amount], as stated in **[Policy Name], [Page Number]**.

            **References:**
            * **[Policy Name 1], [Page Number 1]**
            * **[Policy Name 2], [Page Number 2]**

        Important: Take the policy name and page number from metadata column only
        
        If you cannot find sufficient information to answer the query, indicate that and suggest possible alternative approaches or resources.
        """), ("human", "{question}")]
)


# %%
"""
### RAG pipeline
- Search and rerank
- Generate the context
- Integrate the prompt
- Invoke the LLM
- Parse string output
"""

# %%
"""
**Purpose:**

This code snippet creates a LangChain RAG chain, which combines a retriever function, a prompt, a language model, and an output parser to provide a comprehensive question-answering system.

**Explanation:**

- **`{"context" : results_runnable, "question" : RunnablePassthrough()}`:**
  - This creates a dictionary that defines the components of the RAG chain:
    - `context`: Specifies the `results_runnable` function as the context retriever. This function will be used to retrieve relevant documents based on the query.
    - `question`: Specifies the `RunnablePassthrough` object as the question handler. This object will simply pass the query through without any modification.

- **`| prompt`:**
  - This pipe operator connects the context and question components to a prompt template. The prompt template will be used to format the query and retrieved documents into a suitable input for the language model.

- **`| llm`:**
  - This pipe operator connects the prompt template to the `llm` object, which represents the language model (in this case, the Gemini model). The language model will process the formatted prompt and generate a response.

- **`| StrOutputParser()`:**
  - This pipe operator connects the language model to the `StrOutputParser` object. This parser will ensure that the output from the language model is parsed as a string.

**Impact:**

After running this code, the `rag_chain` variable will contain a RAG chain that can be used to answer questions based on the retrieved documents. When a query is provided to the RAG chain, it will:

1. Retrieve relevant documents using the `results_runnable` function.
2. Format the query and retrieved documents into a prompt using the prompt template.
3. Pass the prompt to the language model.
4. Parse the output from the language model as a string.

The final output will be the generated response from the language model, based on the query and the retrieved documents.

**Note:**

- The specific prompt template used in the RAG chain will determine how the query and retrieved documents are combined into an input for the language model.
- You can customize the RAG chain by modifying the context retriever, prompt template, language model, or output parser.
- The `StrOutputParser` can be replaced with other output parsers if you need to parse the output in a different format.

"""

# %%
rag_chain = ({"context" : results_runnable, "question" : RunnablePassthrough()}
             | prompt
             | llm
             | StrOutputParser()
            )

# %%
"""
### Testing the implementation
The code you provided will retrieve relevant documents from the vector store based on the query "What are the age related conditions in the life insurance?", then rerank the documents using a cross-encoder model, and finally generate a response using the RAG chain.
To get the actual response, you need to run the code and the output will be displayed as Markdown.
"""

# %%
question = "What are the age related conditions in the life insurance?"
result = rag_chain.invoke(question)
display(Markdown(result))

# %%


# %%
"""
--------------------------------
"""

# %%
"""
## **RAG with LlamaIndex - Why Use Llama Index**

Llama Index, a framework designed for building applications powered by large language models (LLMs), offers several key advantages that are evident in the provided code:

#### 1. **Efficient Document Management and Retrieval:**

* **Vector Store Index:** The use of a vector store index allows for efficient storage and retrieval of documents based on their semantic similarity. This enables the system to quickly find relevant information based on the context of the query.
* **Document Loading and Processing:** Llama Index provides tools for loading and processing documents, including text splitting and embedding generation. This simplifies the process of preparing documents for use with the LLM.

#### 2. **Flexible Query Processing:**

* **Retriever and Query Engine:** The `RetrieverQueryEngine` combines a retriever for document retrieval and a response synthesizer for generating responses. This modular architecture allows for customization and flexibility in the query processing pipeline.
* **Prompt Engineering:** Llama Index provides tools for creating and managing prompts, which are the instructions given to the LLM. This allows you to fine-tune the LLM's responses and guide its behavior.

#### 3. **Integration with LLMs:**

* **Seamless Integration:** Llama Index seamlessly integrates with various LLMs, including Gemini. This enables you to leverage the power of these models for tasks like question answering, summarization, and text generation.
* **Customization:** The framework allows you to customize the LLM's behavior through prompt engineering and other techniques.

#### 4. **Extensibility and Customization:**

* **Custom Components:** Llama Index provides a modular architecture that allows you to create custom components, such as retrievers, response synthesizers, or postprocessors, to tailor the system to your specific needs.
* **Integration with Other Tools:** You can easily integrate Llama Index with other tools and libraries to enhance its capabilities.

#### 5. **Performance and Efficiency:**

* **Optimized for Large Datasets:** Llama Index is designed to handle large datasets efficiently, making it suitable for applications that require processing and querying vast amounts of information.

**In summary, Llama Index offers a robust and flexible framework for building applications powered by LLMs. Its efficient document management, flexible query processing, integration with LLMs, extensibility, and performance make it a valuable tool for a wide range of natural language processing tasks.**

"""

# %%
"""
### Import Libraries
**Purpose:**

This code imports necessary libraries for working with language models, embeddings, vector databases, and data analysis. It specifically leverages Llama Index, a framework for building applications powered by language models.

**Key Libraries and Their Functions:**

* **`llama_index.embeddings.huggingface.HuggingFaceEmbedding`:** Provides embeddings using Hugging Face models.
* **`llama_index.llms.gemini.Gemini`:** Interacts with the Gemini language model.
* **`llama_index.core.Settings`:** Configures settings for Llama Index.
* **`llama_index.core.prompts.PromptTemplate`:** Creates prompts for language models.
* **`llama_index.core.node_parser.SentenceSplitter`:** Splits text into sentences.
* **`llama_index.core.VectorStoreIndex`:** Creates a vector store index for storing and retrieving embeddings.
* **`llama_index.core.StorageContext`:** Manages storage for indices.
* **`llama_index.core.get_response_synthesizer`:** Gets a response synthesizer.
* **`llama_index.core.retrievers.VectorIndexRetriever`:** Retrieves documents from a vector index.
* **`llama_index.core.query_engine.RetrieverQueryEngine`:** Creates a query engine using a retriever.
* **`llama_index.core.postprocessor.SimilarityPostprocessor`:** Postprocesses query results based on similarity.
* **`sentence_transformers.CrossEncoder`:** Computes similarity scores between pairs of sentences.
* **`os`:** Provides functions for interacting with the operating system.
* **`IPython.display`:** Enables displaying various types of output, including Markdown, in Jupyter Notebooks.
* **`pandas`:** A powerful library for working with structured data.

**Overall Functionality:**

This code appears to be setting up the environment for building an application that processes text data, extracts information from it, and uses a language model to generate responses. The libraries provide the necessary tools for embedding, indexing, retrieving, and processing text data, as well as interacting with a language model.

**Potential Use Cases:**

* **Question Answering:** Extract information from text documents and answer questions based on their content.
* **Summarization:** Summarize the key points of text documents.
* **Text Generation:** Generate new text based on the content of text documents.
"""

# %%
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

from llama_index.core import Settings

from llama_index.core.prompts import PromptTemplate

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import get_response_synthesizer

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import SimpleDirectoryReader


from sentence_transformers import CrossEncoder

# %%
import os
from IPython.display import display, Markdown
import pandas as pd

# %%
"""
**Purpose:**

This code snippet sets up the environment for using Hugging Face models and the Gemini language model by logging in to Hugging Face, setting the Google API key, and configuring the embedding model and LLM settings in Llama Index.

**Explanation:**

- **`from huggingface_hub import login`:** Imports the `login` function from the `huggingface_hub` library, which is used to log in to Hugging Face.
- **`hf_token = open("API/hf_token", "r").read()`:** Reads the Hugging Face API token from the "API/hf_token" file and stores it in the `hf_token` variable.
- **`login(token = hf_token)`:** Logs in to Hugging Face using the provided API token. This allows you to access and use models from the Hugging Face Hub.
- **`google_api_key = open("API/gemini_key", "r").read()`:** Reads the Google API key from the "API/gemini_key" file and stores it in the `google_api_key` variable.
- **`os.environ["GOOGLE_API_KEY"] = google_api_key`:** Sets the environment variable `GOOGLE_API_KEY` to the value of `google_api_key`. This configures the Gemini language model to use the specified API key.
- **`Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")`:** Sets the `embed_model` setting in Llama Index to the `HuggingFaceEmbedding` class, specifying the model name "BAAI/bge-small-en-v1.5". This will use the specified Hugging Face model for generating embeddings.
- **`Settings.llm = Gemini(model_name="models/gemini-1.5-pro")`:** Sets the `llm` setting in Llama Index to the `Gemini` class, specifying the model name "models/gemini-1.5-pro". This will use the Gemini 1.5 Pro language model for generating responses.

**Impact:**

After running this code, the environment will be configured to use the specified Hugging Face model for embeddings and the Gemini language model for generating responses. You can now use Llama Index to interact with these models and perform tasks like question answering, summarization, and text generation.
"""

# %%
from huggingface_hub import login
hf_token = open("API/hf_token", "r").read()
login(token = hf_token)

# %%
google_api_key = open("API/gemini_key", "r").read()
os.environ["GOOGLE_API_KEY"] = google_api_key

# %%
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") # set the embedding model
Settings.llm = Gemini(model_name="models/gemini-1.5-pro")

# %%


# %%
"""
**Purpose:**

This code snippet loads documents from a specified directory, splits them into sentences, and sets the text splitter in Llama Index settings.

**Explanation:**

- **`documents = SimpleDirectoryReader(input_dir="documents").load_data()`:**
  - Creates an instance of the `SimpleDirectoryReader` class, passing the `input_dir` as an argument.
  - Loads all text files from the specified directory and returns a list of `Document` objects.
  - Stores the list of documents in the `documents` variable.

- **`text_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)`:**
  - Creates an instance of the `SentenceSplitter` class, configuring it with the following parameters:
    - `chunk_size`: The desired size of each text chunk in characters. Here, it's set to 1000.
    - `chunk_overlap`: The number of characters that should overlap between adjacent chunks. Here, it's set to 200.

- **`Settings.text_splitter = text_splitter`:**
  - Sets the `text_splitter` setting in Llama Index to the `text_splitter` object created earlier. This will configure Llama Index to use the specified text splitter for splitting documents into chunks.

**Impact:**

After running this code, the `documents` variable will contain a list of `Document` objects, each representing a text document loaded from the specified directory. The `text_splitter` object will be set as the default text splitter in Llama Index settings.

**Note:**

- The `SimpleDirectoryReader` class can load text files from different formats, such as plain text, Markdown, or PDF.
- The `SentenceSplitter` class can be used to split text into sentences based on various criteria, such as punctuation and sentence structure.
- Setting the `text_splitter` in Llama Index settings ensures that the specified text splitter will be used for all subsequent document splitting operations.

"""

# %%
documents = SimpleDirectoryReader(input_dir="documents").load_data()

# %%
text_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

# global
Settings.text_splitter = text_splitter

# %%
"""
**Purpose:**

This code snippet creates a vector store index from a list of documents, persists the index to storage, and then loads the index from storage for later use.

**Explanation:**

- **`index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])`:**
  - Creates an instance of the `VectorStoreIndex` class, passing the `documents` list and the `text_splitter` as arguments.
  - The `VectorStoreIndex` class stores embeddings of documents and allows for efficient retrieval based on semantic similarity.
  - The `text_splitter` is used to split the documents into smaller chunks before embedding.

- **`index.storage_context.persist(persist_dir="llamaIndex_store")`:**
  - Persists the index to the specified directory "llamaIndex_store". This saves the index to disk for later use.

- **`storage_context = StorageContext.from_defaults(persist_dir="store")`:**
  - Creates a `StorageContext` object from the default settings, specifying the "store" directory as the persist directory.

- **`index = load_index_from_storage(storage_context)`:**
  - Loads the previously persisted index from the storage context. This allows you to reuse the index without having to create it again from scratch.

**Impact:**

After running this code, the `index` variable will contain a `VectorStoreIndex` object that represents the index of the documents. The index is persisted to storage, allowing you to load it later and use it for retrieval and other tasks.

**Note:**

- The `VectorStoreIndex` class uses a vector store to store the embeddings of the documents. The choice of vector store can affect the performance and efficiency of the index.
- The `StorageContext` object provides a way to manage the storage of indices. You can customize the storage settings based on your needs.
- By persisting the index, you can save time by avoiding the need to re-create the index from scratch each time you use it.

"""

# %%
index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])
index.storage_context.persist(persist_dir="llamaIndex_store")

# %%
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="store")

# load index
index = load_index_from_storage(storage_context)

# %%
"""
**Purpose:**

This code snippet creates a prompt template that will be used to format queries and context for a language model.

**Explanation:**

- **`PromptTemplate(template=template, template_var_mappings={"query_str": "question", "context_str": "context"})`:**
  - Creates an instance of the `PromptTemplate` class, passing the following arguments:
    - `template`: A string representing the template for the prompt. This template will contain placeholders for the query and context.
    - `template_var_mappings`: A dictionary that maps variable names in the template to actual values. In this case, the variables "query_str" and "context_str" will be replaced with the actual query and context strings, respectively.

**Impact:**

After running this code, the `prompt_tmpl` variable will contain a `PromptTemplate` object. This object can be used to format queries and context into a suitable input for a language model.


**Note:**

- The `template` string can contain any text you want to include in the prompt, as well as placeholders for the query and context.
- The `template_var_mappings` dictionary can be used to map any variables in the template to their corresponding values.
- The `PromptTemplate` class provides methods for formatting prompts and replacing variables with actual values.

"""

# %%
template = """
You are a knowledgeable and precise assistant specialized in question-answering tasks, 
particularly from academic and research-based sources. 
Your goal is to provide accurate, concise, and contextually relevant answers based on the given information.

Instructions:

Comprehension and Accuracy: Carefully read and comprehend the provided context from the research paper to ensure accuracy in your response.
Conciseness: Deliver the answer in no more than three sentences, ensuring it is concise and directly addresses the question.
Truthfulness: If the context does not provide enough information to answer the question, clearly state, "I don't know."
Contextual Relevance: Ensure your answer is well-supported by the retrieved context and does not include any information beyond what is provided.

Remember if no context is provided please say you don't know the answer
Here is the question and context for you to work with:

\nQuestion: {question} \nContext: {context} \nAnswer:

        **Your Task:**
        1. **Analyze the Query:** Carefully understand the user's intent and the specific information they are seeking.
        2. **Identify Relevant Documents:** Select the most pertinent documents from the search results based on their content and relevance to the query.
        3. **Extract Key Information:** Carefully extract the required information from the selected documents, ensuring accuracy and completeness.
        4. **Construct a Comprehensive Response:** Craft a clear, concise, and informative response that directly addresses the user's query.
        5. **Provide Citations:** Cite the specific policy names and page numbers where the information was found, using the following format:

            **[Policy Name], [Page Number]**

            **References:**
            * [Policy Name 1], [Page Number 1]
            * [Policy Name 2], [Page Number 2]
            * ...

        **Guidelines:**
        * **Accuracy:** Ensure that your response is factually correct and consistent with the information provided in the documents.
        * **Relevance:** Focus on the most relevant information and avoid providing unnecessary details.
        * **Clarity:** Use plain language and avoid technical jargon.
        * **Completeness:** Provide a comprehensive answer that covers all aspects of the user's query.
        * **Conciseness:** Be brief and to the point, while still providing sufficient detail.

        **Example Response:**
        > The maximum coverage for [policy type] is [amount], as stated in **[Policy Name], [Page Number]**.

            **References:**
            * **[Policy Name 1], [Page Number 1]**
            * **[Policy Name 2], [Page Number 2]**

        Important: Take the policy name and page number from metadata column only
        
        If you cannot find sufficient information to answer the query, indicate that and suggest possible alternative approaches or resources.
        """


prompt_tmpl = PromptTemplate(
    template=template,
    template_var_mappings={"query_str": "question", "context_str": "context"},
)

# %%
"""
**Purpose:**

This code snippet configures a retriever, response synthesizer, and query engine for question answering using Llama Index.

**Explanation:**

- **`retriever = VectorIndexRetriever(index=index, similarity_top_k=5)`:**
  - Creates an instance of the `VectorIndexRetriever` class, passing the `index` and `similarity_top_k` as arguments.
  - The `VectorIndexRetriever` retrieves relevant documents from the vector store based on semantic similarity.
  - The `similarity_top_k` parameter specifies the maximum number of documents to retrieve.

- **`response_synthesizer = get_response_synthesizer()`:**
  - Gets a default response synthesizer from Llama Index. This synthesizer can be used to generate responses based on retrieved documents and a prompt.

- **`query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.55)])`:**
  - Creates an instance of the `RetrieverQueryEngine` class, passing the `retriever`, `response_synthesizer`, and `node_postprocessors` as arguments.
  - The `RetrieverQueryEngine` combines the retriever and response synthesizer to provide a complete question-answering system.
  - The `node_postprocessors` list specifies post-processing steps to be applied to the retrieved documents. In this case, a `SimilarityPostprocessor` is used to filter out documents with similarity scores below 0.55.

- **`query_engine.update_prompts({"response_synthesizer:text_qa_template":prompt_tmpl})`:**
  - Updates the prompt template used by the response synthesizer to the `prompt_tmpl` created earlier. This allows you to customize the prompt used for generating responses.

**Impact:**

After running this code, the `query_engine` variable will contain a `RetrieverQueryEngine` object that can be used to answer questions based on the retrieved documents. The `retriever` will retrieve relevant documents from the vector store, the `response_synthesizer` will generate responses based on the retrieved documents and the prompt template, and the `node_postprocessors` will filter out low-quality documents.

**Note:**

- You can customize the `similarity_top_k` parameter to control the number of documents retrieved.
- You can use different response synthesizers or node postprocessors based on your specific requirements.
- The prompt template used in the `query_engine` will determine the format of the prompt that is passed to the language model.
"""

# %%
# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.55)]
)

query_engine.update_prompts(
    {"response_synthesizer:text_qa_template":prompt_tmpl}
)

# %%
"""
### Test the implementation
The code you provided will retrieve relevant documents from the vector store based on the query "What are the cases of failure to pay premium?", rerank the documents using a cross-encoder model, and finally generate a response using the query engine.

To get the actual response, you need to run the code and the output will be displayed as Markdown.
"""

# %%
question = "What are the cases of failure to pay premium?"
result = query_engine.query(question)

# %%
display(Markdown(result.response))

# %%
