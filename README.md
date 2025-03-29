# Retriever + Chart Generator Multi-Agent System

## Overview

This project is a multi-agent system built using `LangGraph` in combination with `Haystack`, where:

- **Retriever Agent**: Uses a **Haystack RAG pipeline** as a tool to fetch relevant information from user-uploaded documents.
- **Chart Generator Agent**: Uses a **Python REPL tool** to generate charts based on retrieved information.

The system follows a structured workflow:
1. **User uploads a document**.
2. **User asks a question**.
3. **Retriever Agent fetches relevant data** utilizing Haystacks' RAG pipelines.
4. **Chart Generator Agent processes the information** and generates a visualization.

![image](https://github.com/user-attachments/assets/ae6ea061-bbc0-405a-92fe-8f7711712768)

---

## Need

### What is an Agent?  
An **agent** is a system that uses an **LLM** to control the flow of an application.  
As complexity increases, managing and scaling agents becomes challenging.  

### Common Problems  
- Too many tools → Leads to poor decision-making  
- Overloaded context → Becomes hard to track  
- Lack of specialization → No dedicated areas like planning, research, or math expertise  

### Solution: Multi-Agent Systems (MAS)  
Breaking applications into smaller, independent agents improves:  

- **Modularity** – Easier development, testing, and maintenance  
- **Specialization** – Agents focused on specific domains  
- **Control** – Explicit agent communication instead of function calling  

### Agents Can Be:  
- A simple **LLM + prompt**
- A complex `ReAct agent`

## Unique Selling Point (USP)  
Unlike many implementations that rely on external API calls, this system runs **entirely offline**.  
- The **LLM model is locally downloaded** and runs on `Ollama`, eliminating the need for external API calls.  
- This ensures **faster responses, reduced costs, and complete data privacy**.  

​
## Features

- **Multi-Agent System** using `LangGraph`  
- **Retrieval-Augmented Generation (RAG) Pipeline** using `Haystack`  
- **Python REPL Execution** for dynamic chart generation  
- **Flask-based API** for document upload and querying  

---

## Setup

### Prerequisites

Ensure you have the following installed:

```bash
pip install flask haystack-ai langchain langchain-core langgraph ollama matplotlib werkzeug
```

---

## System Architecture  

### LLM: `Llama 3.1` (8B) via Ollama  
- Runs **locally** with no API calls.  
- Supports a **context length of 107,362 tokens**.  

### 1. **Retriever Agent**  
- Uses a **Haystack RAG pipeline** to fetch relevant context.  
- Processes document types: `.txt`, `.pdf`, `.md`, `.csv`.  
- Employs **SentenceTransformers** for embeddings (`sentence-transformers/all-MiniLM-L6-v2`).  

### 2. **Chart Generator Agent**  
- Uses a **Python REPL tool** to execute code dynamically.  
- Generates **Matplotlib charts** based on retrieved information.  
- **Ensures all data is hardcoded** within the script for self-contained execution.  

---

## Technologies Used

- **Flask** - Web framework
- **Haystack** - RAG pipeline for document retrieval
- **LangGraph** - Multi-agent workflow execution
- **LangChain** - Agent tool execution
- **Ollama** - Large Language Models
- **Matplotlib** - Chart generation
- **Werkzeug** - Secure file handling

---

## Video Demo

https://github.com/user-attachments/assets/33aaee96-1c1b-43c7-bf80-70b1ae3d2240

## Future Enhancements

- Add support for more file formats.
- Improve retrieval accuracy with better embedding models.

---

## Author

Developed by **Harsh**, AI/ML Enthusiast. 

---
