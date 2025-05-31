import sqlite3
from typing import Annotated
from pydantic import Field, BaseModel

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph, add_messages

import config

# The checkpointer lets the graph persist its state
conn = sqlite3.connect(config.CHECKPOINTS_DIR, check_same_thread=False)
memory = SqliteSaver(conn)

class State(TypedDict):
    question: str
    context: List[Document]
    messages: Annotated[list[AnyMessage], add_messages]
    article_ids: list[str]


class QAGraph:
    """
    Retrieval-generation graph for question answering with chat memory.
    """
    PROMPT = """You are a customer facing assistant tasked with outfit suggestions.
    You will be given descriptions of a number of clothing items that are available in stock. 
    Use this information to provide assistance with attire recommendations based on what's available in stock.
    
    The user request might include an image of a clothing item. In that case, the attached image description will be
    provided together with the user query. 
    
    The output should include clarification questions (if the user's request is not clear) or 
    relevant product recommendations based on the user's request.
    
    User-provided question:
    {question}

    Clothing items available in stock:
    {context}
    """
    LLM = ChatOpenAI(model="gpt-4o")

    def __init__(self, vector_store):
        graph = StateGraph(State).add_sequence([self.decode_message, self.retrieve, self.generate])
        graph.add_edge(START, "decode_message")
        self.graph = graph.compile(checkpointer=memory)
        self.vector_store = vector_store

    def decode_message(self, state: State):
        """
        Decodes the user message and prepares it for retrieval.
        If an image is attached, it generates a description of the image.
        """
        query = state.get('messages').pop()
        text_element = next((item for item in query.content if item.get("type") == "text"), None)
        image_element = next((item for item in query.content if item.get("type") == "image"), None)

        decoded_message = text_element.get('text', '') if text_element else ''

        if image_element:
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the clothing item(s) in this image:",
                    },
                    image_element,
                ],
            }
            image_description = self.LLM.invoke([message])
            decoded_message += f"\n\nAttached image description: {image_description.content}"


        return {"messages": HumanMessage(content=decoded_message)}

    def retrieve(self, state: State):
        """Retrieval step responsible for retrieving the top-k relevant documents from the vector store"""
        user_query = state["messages"][-1].content
        retrieved_docs = self.vector_store.similarity_search(user_query, k=2)
        return {"context": retrieved_docs}

    def generate(self, state: State):
        """Generation step responsible for generating an answer to the question provided as input"""
        docs_content = self.format_documents(state["context"])
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.PROMPT,
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(context=docs_content, question=state["messages"][-1].content)
        assistant_runnable = prompt_template | self.LLM.with_structured_output(schema=FashionRecommenderOutput)
        response = assistant_runnable.invoke(state)
        return {"messages": AIMessage(content=response.message), "article_ids": response.article_ids}

    @staticmethod
    def format_documents(documents: List[Document]) -> str:
        """
        Formats the retrieved documents into a string for prompting
        """
        return "\n\n".join(
            f"Product_id: <{doc.id}>\n"
            f"Product name: {doc.metadata.get('product_name')}\n"
            f"Product type: {doc.metadata.get('product_type_name')}\n"
            f"Product colour: {doc.metadata.get('colour_group_name')}\n"
            f"Description: {doc.page_content}"
            for doc in documents
        )


class FashionRecommenderOutput(BaseModel):
    message: str = Field(description="Response message from the assistant to the user")
    article_ids: list[str] = Field(
        default=[],
        description="List of article IDs recommended by the assistant. "
                    "Only populate this if the assistant is making product recommendations."
                    "Otherwise, return empty list if the assistant is providing additional details about the products"
                    "already recommended.")
