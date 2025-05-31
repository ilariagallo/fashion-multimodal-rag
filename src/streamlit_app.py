import streamlit as st
import base64
import uuid
import os

from langchain_core.messages import HumanMessage
from indexing import MultiModalIndex
from qa_graph import QAGraph
import config

st.set_page_config(page_title="Chat with Text & Image", layout="centered")
st.title("🛍️✨Chat & Choose Your Outfit!")

# ---------------- Caching Heavy Resources ----------------
@st.cache_resource
def get_articles_db():
    return MultiModalIndex()


@st.cache_resource
def get_qa_graph(_articles_db):
    return QAGraph(_articles_db.vector_store)


class ChatApp:
    """
    A Streamlit application that allows users to chat with an AI assistant,
    providing both text and image inputs, and receiving responses with relevant article images.
    """

    def __init__(self):
        self.thread_id = self.initialize_session_state()
        self.articles_db = get_articles_db()
        self.qa_graph = get_qa_graph(self.articles_db)

    @staticmethod
    def initialize_session_state():
        """Initialize the session state to store chat messages if not already present."""
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = str(uuid.uuid4())
        if "messages" not in st.session_state:
            st.session_state.messages = []
        return st.session_state.thread_id

    @staticmethod
    def display_messages():
        """Render all chat messages (text and images) stored in session state."""
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("text"):
                    st.markdown(msg["text"])
                if msg.get("image"):
                    st.image(msg["image"], caption="Uploaded image", width=300)
                if msg.get("article_images"):
                    for img_path in msg["article_images"]:
                        st.image(img_path, caption=os.path.basename(img_path), width=300)

    @staticmethod
    def get_user_input():
        """
        Display the input form for the user to enter a message and optionally upload an image.

        Returns:
            tuple: (user_input: str, uploaded_image: UploadedFile, submitted: bool)
        """
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([3, 1])
            user_input = col1.text_input("Your message:", label_visibility="collapsed")
            uploaded_image = col2.file_uploader(" ", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
            submitted = st.form_submit_button("Send")
        return user_input, uploaded_image, submitted

    @staticmethod
    def encode_uploaded_image(uploaded_image):
        """
        Encode an uploaded image file to a base64 string for processing.

        Args:
            uploaded_image (UploadedFile): Image file uploaded by user.

        Returns:
            dict: Encoded image content in expected format.
        """
        image_bytes = uploaded_image.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "type": "image",
            "source_type": "base64",
            "data": encoded_image,
            "mime_type": "image/jpeg",
        }

    @staticmethod
    def get_article_image_paths(article_ids, base_path=config.DATA_DIR + "images"):
        """
        Given a list of article IDs, return the file paths of corresponding image files.

        Args:
            article_ids (list): List of article ID strings.
            base_path (str): Directory where article images are stored.

        Returns:
            list: List of valid image file paths.
        """
        image_paths = []
        for article_id in article_ids:
            image_path = os.path.join(base_path, f"{article_id}.jpg")
            if os.path.exists(image_path):
                image_paths.append(image_path)
        return image_paths

    @staticmethod
    def append_user_message(user_input, uploaded_image):
        image_bytes = uploaded_image.read() if uploaded_image else None
        st.session_state.messages.append({
            "role": "user",
            "text": user_input if user_input else None,
            "image": image_bytes
        })

    @staticmethod
    def append_assistant_message(ai_message, image_paths):
        """
        Append the assistant's response and related article images to the session state.

        Args:
            ai_message (str): Assistant's textual response.
            image_paths (list): List of image paths to be shown with the response.
        """
        st.session_state.messages.append({
            "role": "assistant",
            "text": ai_message,
            "article_images": image_paths
        })

    def build_message_content(self, user_input, uploaded_image):
        """
        Build a list of message content blocks (text and image) for the assistant.

        Args:
            user_input (str): The user's message.
            uploaded_image (UploadedFile): An optional uploaded image.

        Returns:
            list: Formatted message content suitable for model input.
        """
        content = [{"type": "text", "text": user_input}]
        if uploaded_image:
            content.append(self.encode_uploaded_image(uploaded_image))
        return content

    def generate_bot_response(self, message_content):
        """
        Generate the assistant's response using the QA graph and message content.

        Args:
            message_content (list): A list of text/image content blocks.

        Returns:
            tuple: Assistant's textual response and list of article IDs referenced.
        """
        messages = [HumanMessage(content=message_content)]
        conf = {"configurable": {"thread_id": self.thread_id}}  # Unique thread ID
        response = self.qa_graph.graph.invoke({"messages": messages}, conf)
        return response['messages'][-1].content, response['article_ids']

    def process_user_message(self, user_input, uploaded_image):
        """
        Handle full interaction flow: process user message, call model, and update UI.

        Args:
            user_input (str): User's text input.
            uploaded_image (UploadedFile): Optional image uploaded with message.
        """
        message_content = self.build_message_content(user_input, uploaded_image)

        self.append_user_message(user_input, uploaded_image)

        ai_message, article_ids = self.generate_bot_response(message_content)
        image_paths = self.get_article_image_paths(article_ids)

        self.append_assistant_message(ai_message, image_paths)

        st.rerun()


    # ---------------- Main App ----------------
    def run(self):
        """Run the Streamlit app."""
        self.display_messages()

        user_input, uploaded_image, submitted = self.get_user_input()
        if submitted:
            self.process_user_message(user_input, uploaded_image)


# ---------------- Entry point ----------------
if __name__ == "__main__":
    app = ChatApp()
    app.run()