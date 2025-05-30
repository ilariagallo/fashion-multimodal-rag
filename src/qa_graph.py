from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph, add_messages

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()

class State(TypedDict):
    question: str
    context: List[Document]
    messages: Annotated[list[AnyMessage], add_messages]


class QAGraph:
    """
    Retrieval-generation graph for question answering
    """
    PROMPT = """You are a customer facing assistant tasked with outfit suggestions.
    You will be given descriptions of a number of clothing items that are available in stock. 
    Use this information to provide assistance with attire recommendations based on what's available in stock.
    The output should include clarification questions (if the user's request is not clear) or relevant product 
    recommendations based on the user's request.

    User-provided question:
    {question}

    Clothing items available in stock:
    {context}
    """
    LLM = ChatOpenAI(model="gpt-4o")

    def __init__(self, vector_store):
        graph = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph.add_edge(START, "retrieve")
        self.graph = graph.compile(checkpointer=memory)
        self.vector_store = vector_store

    def retrieve(self, state: State):
        """Retrieval step responsible for retrieving the top-k relevant documents from the vector store"""
        user_query = state["messages"][-1].content
        retrieved_docs = self.vector_store.similarity_search(user_query, k=2)
        return {"context": retrieved_docs}

    def generate(self, state: State):
        """Generation step responsible for generating an answer to the question provided as input"""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.PROMPT,
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(context=docs_content, question=state["messages"][-1].content)
        assistant_runnable = prompt_template | self.LLM
        response = assistant_runnable.invoke(state)
        return {"messages": AIMessage(content=response.content)}
