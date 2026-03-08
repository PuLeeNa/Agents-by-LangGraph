from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage 
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

document_content = ""

class AgenState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """Updates the document content with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully!, The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the current documrnt to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"Document has been saved successfully as {filename}!")
        return f"Document has been saved successfully as {filename}!"
    
    except Exception as e:
        return f"An error occurred while saving the document: {str(e)}"
    

tools = [update, save]

model = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools)

def our_agent(state: AgenState) -> AgenState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the use update and modify documents.
                                  
    - If the user wants to update the document, use the 'update' tool and pass the new content as an argument.
    - If the user wants to save the document, use the 'save' tool and pass the desired filename as an argument. After saving the document, you should end the process.
    - Make sure to always show the current document state after modifications.
                                  
    The current document content is:{document_content}                
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like me to create?"
        user_message = HumanMessage(content=user_input)

    else: 
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\nUser: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgenState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]

    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and  "saved" in message.content.lower() and "document" in message.content.lower():
            return "end"
        
    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\nTOOL RESULT: {message.content}")

graph = StateGraph(AgenState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    }
)

app = graph.compile()

# START -> agent 
#           ^ |
#           | v
#          tools -> END

def run_document_agent():
    print("\n ===== DRAFTER =====")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== END DRAFTER =====")

if __name__ == "__main__":
    run_document_agent()