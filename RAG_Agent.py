from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage 
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

pdf_path = "GEN_AI.pdf"

pdf_loader = PyPDFLoader(pdf_path)

pages = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

pages_split = text_splitter.split_documents(pages)

persist_directory = "pdf_embeddings"
collection_name = "gen_ai"


try:
    vectorestore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Created ChromaDB vector store!")

except Exception as e:
    print(f"An error occurred while creating the vector store: {str(e)}")
    raise


retreiver = vectorestore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the PDF file based on the user's query.
    """
    docs = retreiver.invoke(query)

    if not docs:
        return "No relevant information found in the document."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tool)

class AgenState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgenState):
    """Determines whether the agent should continue or end the process based on the last message's tool calls."""
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
system_prompt = """
You are a helpful assistant designed to answer questions based on the content of a PDF document. You have access to a tool called 'retriever_tool' that allows you to search for relevant information within the PDF. 
When you receive a query, use the 'retriever_tool' to find and retrieve the most relevant information from the PDF. If the retrieved information is sufficient to answer the user's question, provide a clear and concise response. If additional information is needed, continue using the 'retriever_tool' until you have enough information to answer the query effectively. 
Always ensure that your responses are based on the content of the PDF and provide accurate information to the user.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}

# LLM Agent
def call_llm(state: AgenState) -> AgenState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": message}



