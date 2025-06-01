import uuid

from langchain_core.runnables import RunnableLambda

from langsmith.schemas import Example, Run
from langsmith import evaluate

from src.indexing import MultiModalIndex
from src.qa_graph import QAGraph

"""
This module offers evaluators to measure the relevance of the recommendations.
Evaluation metrics will be recorded in LangSmith.
"""

# ----- DEFINE EVALUATORS -----

def is_relevant_recommendation(root_run: Run, example: Example) -> dict:
    """Evaluator to check if the article IDs in the response are relevant to the expected output

    :param root_run: The run containing the actual outputs
    :param example: The example containing the expected outputs
    :return: A dictionary with the evaluation result
    """
    return is_relevant(root_run, example, 'article_ids')

def is_relevant(root_run: Run, example: Example, field_name: str) -> dict:
    """
    Evaluator to check if a specific field form the response is relevant to the expected output
    :param root_run: The run containing the actual outputs
    :param example: The example containing the expected outputs
    :param field_name: The name of the field to check for relevance (e.g., 'article_ids', 'text', etc.)
    :return: A dictionary with the evaluation result
    """
    root_run_outputs = root_run.outputs
    example_outputs = example.outputs

    expected_output = example_outputs.get(field_name)
    actual_output = root_run_outputs.get(field_name)

    is_response_relevant = expected_output == actual_output

    return {"key": "is_" + field_name + "_relevant", "score": is_response_relevant}

# ----- EVALUATION FUNCTION -----

def run_evaluate(dataset_name, experiment_name):
    """ Run the evaluation of the QAGraph against a dataset."""
    return evaluate(
        wrapped_graph,
        data= dataset_name,
        evaluators=[is_relevant_recommendation],
        experiment_prefix=experiment_name,
    )


def wrapped_invoke(input_msg):
    """Invoke the QAGraph with a given input message. This function is wrapped to support "config".

    :param input_msg: The input message to be processed by the QAGraph
    :return: The response from the QAGraph
    """
    index = MultiModalIndex()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    qa_graph = QAGraph(index.vector_store)
    return qa_graph.graph.invoke(input_msg, config=config)

wrapped_graph = RunnableLambda(wrapped_invoke)


if __name__ == "__main__":
    run_evaluate(dataset_name='fashion-dataset', experiment_name='test-experiment')
