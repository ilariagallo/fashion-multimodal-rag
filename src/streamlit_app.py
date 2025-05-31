import streamlit as st
import base64
import uuid

from langchain_core.messages import HumanMessage
from indexing import MultiModalIndex
from qa_graph import QAGraph


def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("text"):
                st.markdown(msg["text"])
            if msg.get("image"):
                st.image(msg["image"], caption="Uploaded image", width=300)


def get_user_input():
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        user_input = col1.text_input("Your message:", label_visibility="collapsed")
        uploaded_image = col2.file_uploader(" ", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        submitted = st.form_submit_button("Send")
    return user_input, uploaded_image, submitted


def generate_bot_response(message_content):
    articles_db = MultiModalIndex()
    qa_graph = QAGraph(articles_db.vector_store)
    messages = [HumanMessage(content=message_content)]
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # Unique thread ID for checkpoints
    response = qa_graph.graph.invoke({"messages": messages}, config)
    return response['messages'][-1].content


def process_user_message(user_input, uploaded_image):
    message_content = [{"type": "text", "text": user_input}]

    if uploaded_image:
        image_bytes = uploaded_image.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        message_content.append(
            {
                "type": "image",
                "source_type": "base64",
                "data": encoded_image,
                "mime_type": "image/jpeg",
            }
        )

    # Append user message to session state
    st.session_state.messages.append({
        "role": "user",
        "text": user_input if user_input else None,
        "image": uploaded_image if uploaded_image else None
    })

    # Get bot response
    ai_message = generate_bot_response(message_content)

    # Append assistant message to session state
    st.session_state.messages.append({
        "role": "assistant",
        "text": ai_message
    })

    st.rerun()


def main():
    st.set_page_config(page_title="Chat with Text & Image", layout="centered")
    st.title("🛍️✨Chat & Choose Your Outfit!")

    initialize_session_state()
    display_messages()

    user_input, uploaded_image, submitted = get_user_input()

    if submitted:
        process_user_message(user_input, uploaded_image)


if __name__ == "__main__":
    main()
