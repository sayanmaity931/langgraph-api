from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


app = FastAPI(title="LangGraph Chat API")

load_dotenv()

model = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    task="text-generation"
)

llm = ChatHuggingFace(llm = model)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect("chat_memory.db", check_same_thread=False) 
checkpointer = SqliteSaver(conn) ## u can use InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

class ChatRequest(BaseModel):
    thread_id: str
    message: str
    user_id: str | None = None  # optional (future use)

class ChatResponse(BaseModel):
    thread_id: str
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {
        "configurable": {
            "thread_id": req.thread_id
        }
    }

    result = chatbot.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config
    )

    ai_msg = result["messages"][-1]

    print(ai_msg)

    return ChatResponse(
        thread_id=req.thread_id,
        response=ai_msg.content
    )
 
# but this is not production ready