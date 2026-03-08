from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Messages for providing instructions to the LLM
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()


# Reducer Function (add_messages)
# Rule that controls how updates from nodes are combined with the existing state.
# Tells us how to merge new data into the current state
# without a reducer, updates would have replaced the existing value entirely

# without a reducer
state = {"messages": ["Hi!"]}
update = {"messages": ["How are you?"]}
new_state = {"messages": ["How are you?"]} # the new value replaces the old one

# with a reducer
state = {"messages": ["Hi!"]}
update = {"messages": ["How are you?"]}
new_state = {"messages": ["Hi!", "How are you?"]} # the new value is combined with the old one



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """A simple tool that adds two numbers together."""
    return a + b

@tool
def subtract(a: int, b: int):
    """A simple tool that subtracts two numbers."""
    return a - b

@tool
def multiply(a: int, b: int):
    """A simple tool that multiplies two numbers."""
    return a * b

tools = [add, subtract, multiply]

model = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
            "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

# START -> our_agent -> END
#           ^ |
#           | v
#          tools


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# inputs = {"messages": [("user", "Add 3 + 4.")]}
# print_stream(app.stream(inputs, stream_mode="values"))



#   (.venv) PS C:\26\AI\LangGraph> python .\ReAct.py
#   ================================ Human Message =================================

#   Add 3 + 4.
#   ================================== Ai Message ==================================
#   Tool Calls:
#       add (9x93tafbe)
#   Call ID: 9x93tafbe
#       Args:
#       a: 3
#       b: 4
#   ================================= Tool Message =================================
#   Name: add

#   7
#   ================================== Ai Message ==================================

#   The result of the addition is 7.


inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 2 and then subtract 5")]}
print_stream(app.stream(inputs, stream_mode="values"))

    # (.venv) PS C:\26\AI\LangGraph> python .\ReAct.py
    # ================================ Human Message =================================

    # Add 40 + 12 and then multiply the result by 2 and then subtract 5
    # ================================== Ai Message ==================================
    # Tool Calls:
    # add (18qrp1wxs)
    # Call ID: 18qrp1wxs
    # Args:
    #     a: 40
    #     b: 12
    # multiply (prnp44n4m)
    # Call ID: prnp44n4m
    # Args:
    #     a: 52
    #     b: 2
    # subtract (192ztytf7)
    # Call ID: 192ztytf7
    # Args:
    #     a: 104
    #     b: 5
    # ================================= Tool Message =================================
    # Name: subtract

    # 99
    # ================================== Ai Message ==================================

    # The result of the operations is 99.