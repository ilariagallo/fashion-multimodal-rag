import uuid

from langchain_core.messages import HumanMessage

from src.indexing import MultiModalIndex
from src.qa_graph import QAGraph

if __name__ == "__main__":
    index = MultiModalIndex()

    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    qa_graph = QAGraph(index.vector_store)
    conversation_ongoing = True
    while conversation_ongoing:
        user_input = input("\n👤 User:\n")
        messages = [HumanMessage(content=user_input)]
        response = qa_graph.graph.invoke({"messages": messages}, config)
        ai_message = response['messages'][-1]

        # Display output
        print("\n🤖 Assistant:\n", ai_message.content)