import asyncio
import sqlite3
from typing import Annotated
from pydantic import Field, BaseModel

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph, add_messages
from langsmith import utils

from config import *
from prompts import *


# LangSmith utilities for tracing
utils.tracing_is_enabled()

# Define the state of the graph
class State(TypedDict):
    question: str
    context: List[Document]
    messages: Annotated[list[AnyMessage], add_messages]
    article_ids: list[str]

# The checkpointer lets the graph persist its state
conn = sqlite3.connect(CHECKPOINTS_DIR, check_same_thread=False)
memory = SqliteSaver(conn)


class QAGraph:
    """
    Retrieval-generation graph for question answering with chat memory and guardrails.
    """
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
                        "text": IMAGE_DESCRIPTION_PROMPT,
                    },
                    image_element,
                ],
            }
            image_description = chat.invoke([message])
            decoded_message += f"\n\nAttached image description: {image_description.content}"


        return {"messages": HumanMessage(content=decoded_message)}

    def retrieve(self, state: State):
        """Retrieval step responsible for retrieving the top-k relevant documents from the vector store"""
        latest_chat_history = "\n\n".join([f'{msg.type.upper()}: {msg.content}' for msg in state["messages"][-3:]])
        retrieved_docs = self.vector_store.similarity_search(latest_chat_history, k=7)
        return {"context": retrieved_docs}

    def generate(self, state: State):
        """Generation step responsible for generating an answer to the question provided as input"""
        docs_content = self.format_documents(state["context"])
        user_message = state["messages"][-1].content
        response_message, articles_ids = asyncio.run(
            self.execute_chat_with_guardrail_async(user_message, docs_content, state)
        )
        return {"messages": AIMessage(content=response_message), "article_ids": articles_ids}

    @staticmethod
    def format_documents(documents: List[Document]) -> str:
        """
        Formats the retrieved documents into a string for prompting

        :param documents: List of Document objects retrieved from the vector store
        :return: Formatted string with product details
        """
        return "\n\n".join(
            f"Product_id: <{doc.id}>\n"
            f"Product name: {doc.metadata.get('product_name')}\n"
            f"Product type: {doc.metadata.get('product_type_name')}\n"
            f"Product colour: {doc.metadata.get('colour_group_name')}\n"
            f"Description: {doc.page_content}"
            for doc in documents
        )


    # ---- FUNCTIONALITY TO GENERATE CHAT RESPONSE WITH INPUT GUARDRAILS ---- #

    async def execute_chat_with_guardrail_async(self, user_message, docs_content, state) -> str:
        """ Asynchronously executes the chat with a topical guardrail to ensure the conversation stays on topic."""
        return await self.execute_chat_with_guardrail(user_message, docs_content, state)

    async def get_chat_response(self, user_request, context, state: State):
        """
        Generates a chat response using the LLM and the provided context.

        :param user_request: The user's request message.
        :param context: The context retrieved from the vector store.
        :param state: The current state of the conversation.
        :return: The chat response and the article IDs recommended by the assistant.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    QA_PROMPT
                ),
                ("placeholder", "{messages}"),
            ]
        ).partial(context=context, question=user_request)
        assistant_runnable = prompt_template | chat.with_structured_output(schema=FashionRecommenderOutput)
        response = assistant_runnable.invoke(state)
        return response.message, response.article_ids

    async def topical_guardrail(self, user_request):
        """
        Checks if the user request is within the allowed topics using a topical guardrail.
        If the topic is allowed, it returns 'allowed', otherwise it returns 'not_allowed'.

        :param user_request: The user's request message.
        :return: 'allowed' or 'not_allowed' based on the topic of the user request.
        """
        messages = [
            {
                "role": "system",
                "content": GUARDRAIL_PROMPT,
            },
            {"role": "user", "content": user_request},
        ]
        response = chat.invoke(messages)
        return response.content

    async def execute_chat_with_guardrail(self, user_request, context, state: State):
        """
        Executes the chat with a topical guardrail to ensure the conversation stays on topic.
        If the guardrail is triggered, it returns a predefined response.

        :param user_request: The user's request message.
        :param context: The context retrieved from the vector store.
        :param state: The current state of the conversation.
        :return: The chat response or a guardrail-triggered response.
        """
        topical_guardrail_task = asyncio.create_task(self.topical_guardrail(user_request))
        chat_task = asyncio.create_task(self.get_chat_response(user_request, context, state))

        while True:
            done, _ = await asyncio.wait(
                [topical_guardrail_task, chat_task], return_when=asyncio.FIRST_COMPLETED
            )
            if topical_guardrail_task in done:
                guardrail_response = topical_guardrail_task.result()

                # Topical guardrail triggered
                if guardrail_response == "not_allowed":
                    chat_task.cancel()
                    response_message = GUARDRAIL_SAFE_RESPONSE
                    return response_message, _

                # Topical guardrail passed
                elif chat_task in done:
                    chat_response = chat_task.result()
                    return chat_response
            else:
                await asyncio.sleep(0.1)  # sleep for a bit before checking the tasks again


class FashionRecommenderOutput(BaseModel):
    """
    Structured output model for the fashion recommender assistant's response.
    """
    message: str = Field(description="Response message from the assistant to the user")
    article_ids: list[str] = Field(
        default=[],
        description="List of article IDs recommended by the assistant. "
                    "Only populate this if the assistant is making product recommendations or if the user explicitly "
                    "asks to see the product. Otherwise, return empty list if the assistant is providing additional "
                    "details about the products already recommended. This is used to retrieve product images for "
                    "display in the UI.")
