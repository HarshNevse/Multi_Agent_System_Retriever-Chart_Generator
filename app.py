import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.routers import FileTypeRouter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.joiners import DocumentJoiner
from haystack.components.converters.csv import CSVToDocument
from haystack import component, Document
from typing import List, Annotated, Literal
import re

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langgraph.graph import StateGraph, START

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"txt", "pdf", "md", "csv"}

# Initialize Document Store
document_store = InMemoryDocumentStore()

# File Processing Pipelines
file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown", "text/csv"])
text_file_converter = TextFileToDocument()
markdown_converter = MarkdownToDocument()
pdf_converter = PyPDFToDocument()
csv_converter = CSVToDocument()
document_joiner = DocumentJoiner()

document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(split_by="word", split_length=250, split_overlap=10)

document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_writer = DocumentWriter(document_store)

preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
preprocessing_pipeline.add_component(instance=csv_converter, name="csv_converter")
preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
preprocessing_pipeline.connect("file_type_router.text/csv", "csv_converter.sources")
preprocessing_pipeline.connect("text_file_converter", "document_joiner")
preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
preprocessing_pipeline.connect("markdown_converter", "document_joiner")
preprocessing_pipeline.connect("csv_converter", "document_joiner")
preprocessing_pipeline.connect("document_joiner", "document_splitter")
preprocessing_pipeline.connect("document_splitter", "document_embedder")
preprocessing_pipeline.connect("document_embedder", "document_writer")


from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
rag_pipe = Pipeline()
rag_pipe.add_component("embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
rag_pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=40))

rag_pipe.connect("embedder.embedding", "retriever.query_embedding")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "files" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist("files")

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                # Process the uploaded file
                preprocessing_pipeline.run({"file_type_router": {"sources": [filepath]}})

        return jsonify({"message": "Files uploaded and processed successfully!"})

    return render_template("index.html")


# MULTI-AGENT SYSTEM

repl = PythonREPL()

@tool
def retrieval_tool(query: Annotated[str, "Question asked by the user to search context in the retriever."]):
    '''Use this to retreive relevant context to answer the user's question. The better the input question is framed, 
        the better the semantically retrieved context is.'''
    print(f"Retrieval tool called with query: {query}\nRunning RAG pipeline...")
    response = rag_pipe.run({"embedder": {"text": query}})
    
    result = '\n'.join([doc.content for doc in response["retriever"]["documents"]])
    print("Retrieval Tool successfully used.")

    return f"Retrieved Context:\n{result}"

import matplotlib.pyplot as plt

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    print(f"repl tool invoked with code: \n {code}")
    try:
        # Ensure 'static' directory exists
        os.makedirs("static", exist_ok=True)

        # Remove plt.show() to avoid GUI issues
        code = code.replace("plt.show()", "")

        # Ensure figure is saved properly
        plot_path = "static/chart.png"
        code += f"\nplt.savefig('{plot_path}', bbox_inches='tight')"

        result = repl.run(code)

        return {
            "message": "Successfully executed.",
            "code": f"```python\n{code}\n```",
            "stdout": result + 'If you have completed the task, respond with FINAL ANSWER.',
            "plot_path": plot_path,  # Returning path to access from frontend
        }

    except Exception as e:
        print(f"Error in python code: \n{code}, \nerror is: \n{repr(e)}")
        return {"error": f"Failed to execute. Error: {repr(e)}"}


def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )

llm = ChatOllama(model="llama3.1:8b", temperature=0.1, num_gpu=1)

def get_next_node(last_message: BaseMessage, goto: str):
    print(f"Checking if '{last_message.content}' contains 'FINAL ANSWER'...")
    if "FINAL ANSWER" in last_message.content:
        print("Final answer found. Ending process.")
        return END
    print(f"No final answer. Moving to: {goto}")
    return goto


research_agent = create_react_agent(llm, tools=[retrieval_tool], prompt=make_system_prompt('''You can ONLY do retrieval. You are working with a chart generator colleague. After you get the context, YOU CANNOT GENERATE PYTHON CHART, SEND INFORMATION TO CHART GENERATOR AFTER YOU HAVE REASONED IT. send the relevant information required to the chart generator for relevant use according to the user question. REMEMBER:  Send the information in natural language, DO NOT send it in python code.'''))
chart_agent = create_react_agent(llm, [python_repl_tool], prompt=make_system_prompt('''You can only generate python plots by writing python code. You are working with a researcher colleague. MAKE SURE every data is hardcoded in your python code, **DO NOT** to access any variable from the researcher, MAKE YOUR OWN VARIABLES IN YOUR PYTHON CODE SCOPE. MAKE SURE TO STRICTLY DEFINE ALL VARIABLES IN THE SCOPE OF THE PYTHON SCRIPT.
IMPORTANT: ONCE YOU MAKE THE CHART, FINISH AND STOP. '''))

def research_node(state: MessagesState) -> Command[Literal["chart_generator", END]]:
    print("Research Agent Invokedd...")
    result = research_agent.invoke(state)
    print("Research Agent Finished Processing...")
    goto = get_next_node(result["messages"][-1], "chart_generator")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="researcher"
    )
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    print("Chart agent invoked...", flush = True)
    result = chart_agent.invoke(state)
    print("Chart agent finished...", flush = True)
    goto = get_next_node(result["messages"][-1], "researcher")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )

workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_edge(START, "researcher")
graph = workflow.compile()

@app.route("/query", methods=["POST"])
def query():
    query_text = request.json.get("query")
    response = graph.invoke(input = {"messages": query_text})
    final_output = ''
    for msg in response['messages']:
        if msg.content:
            final_output += '---'*10 + '\n' + msg.content + '\n'
    
    return jsonify({"results": final_output})

if __name__ == "__main__":
    app.run(debug=True, use_reloader = False)
