import json
import random
from typing import Optional

from distilabel.pipeline import Pipeline
from distilabel.steps import (
    KeepColumns,
    LoadHubDataset,
    StepInput,
    StepOutput,
    step,
)
from distilabel.steps.tasks import (
    TextGeneration,
)
from huggingface_hub import DatasetCard, InferenceClient, get_token
from pydantic import BaseModel, conlist, constr

from custom_llm import InferenceEndpointsLLMWithGrammar

EMBEDDING_MODEL_ENDPOINT_URL = (
    "https://tmu6gkvjx3vvppfl.us-east-1.aws.endpoints.huggingface.cloud"
)

INPUT_DATASET_ID = "davanstrien/self-oss-instruct-sc2-exec-filter-50k-short"
OUTPUT_DATASET_ID = "davanstrien/similarity-dataset-sc2-8b"
NUM_EXAMPLES = None  # to run all
MODEL_ID = None
TEXT_COLUMN_NAME = "instruction"
END_POINT_NAME = "meta-llama-3-8b-instruct-aeu"


class Prompts(BaseModel):
    good: conlist(constr(min_length=100), min_length=2, max_length=2)  # type: ignore
    bad: conlist(constr(min_length=100), min_length=2, max_length=2)  # type: ignore


schema = Prompts.model_json_schema()


@step(
    inputs=["generation", "instruction"],
    outputs=["positive", "negative"],
)
def mine_hard_negative(inputs: StepInput) -> StepOutput:
    """Mine hard negative examples for the generation."""
    # Initialize the inference client
    client = InferenceClient(
        model=EMBEDDING_MODEL_ENDPOINT_URL,
        token=get_token(),
    )
    clean = []
    for input in inputs:
        try:
            original_text = input["instruction"]
            data = json.loads(input["generation"])
            # Validate the data matches the schema
            try:
                _ = Prompts(**data)
            except Exception:
                # Skip the input if it doesn't match the schema
                continue
            # Select a random positive example
            positive = random.choice(data["good"])
            negative_candidates = data["bad"]
            # Find the most similar negative example
            embeddings = client.sentence_similarity(
                original_text, negative_candidates
            ).get("similarities")
            most_similar = negative_candidates[embeddings.index(max(embeddings))]
            negative = most_similar
            input["positive"] = positive
            input["negative"] = negative
            clean.append(input)
        except Exception as e:
            print(e)
            continue
    yield clean


def format_prompt(text: str) -> str:
    return f"""
Here is a natural language prompt from a user for writing Python code: "{text}"

Task:
Your role is to rewrite this prompt to create both similar and dissimilar examples.

1. Generate 2 'good' examples where the prompt has the same meaning and intent but is phrased differently. 
   - Vary the phrasing, terminology, or structure while preserving the original meaning.
   - The functions resulting from the rephrased prompts should pass the same test cases as the original.

2. Generate 2 'bad' examples where the prompt command significantly changes in meaning or intent.
   - The changes should be substantial enough to alter what an appropriate Python function would do.
   - The functions returned from the response to the rephrased prompts should fail the test cases of the original.

Additional guidelines:
- The length of the generated examples should be similar to the original text.
- Ensure the 'bad' examples are reasonable prompts, but with a different meaning or intent.

Return your examples as a JSON object with the keys 'good' and 'bad', and the rewritten prompts as an array. Use the following JSON schema:

{schema}
"""


def update_card(dataset_id):
    try:
        card = DatasetCard.load(dataset_id)
        if "sentence-transformers" not in card.data.tags:
            card.data.tags.append("sentence-transformers")
        if "DistilSimData" not in card.data.tags:
            card.data.tags.append("DistilSimData")
        card.push_to_hub(dataset_id)
    except Exception as e:
        print(e)


@step(inputs=["text"], outputs=["instruction"])
def format_prompts(inputs: StepInput) -> StepOutput:
    """Format the input text into a prompt for the LLM."""
    for input in inputs:
        input["instruction"] = format_prompt(input["text"])
    yield inputs


def create_pipeline(
    model_id: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    text_column_name: Optional[str] = None,
    llm_inference_batch_size=10,
) -> Pipeline:
    """Create a pipeline for generating paraphrases and mining hard negative examples."""
    # Define the API key once since it is used in multiple places.
    api_key = get_token()
    # Initialize the LLM
    llm = InferenceEndpointsLLMWithGrammar(
        endpoint_name=endpoint_name or None,
        model_id=None if endpoint_name else model_id,
        api_key=api_key,
    )
    # Define the column name mapping for the dataset.
    column_name_mapping = (
        {text_column_name: "text"} if text_column_name else {"text": "text"}
    )

    # Assemble the pipeline using a with statement.
    with Pipeline(
        name="create-embeddings",
        description="Create embeddings for text data",
    ) as pipeline:
        load_data = LoadHubDataset(
            name="load_dataset",
            output_mappings=column_name_mapping,
        )
        format_input = format_prompts(name="format_input")
        text_generation = TextGeneration(
            name="paraphrase_text",
            llm=llm,
            input_batch_size=llm_inference_batch_size,
        )
        select_sentences = mine_hard_negative(name="select_sentences")
        columns_to_keep = KeepColumns(
            columns=["text", "positive", "negative", "generation"],
            output_mappings={"text": "anchor"},
        )
        # assemble the pipeline
        (
            load_data
            >> format_input
            >> text_generation
            >> select_sentences
            >> columns_to_keep
        )

    return pipeline


def run_pipeline(
    dataset_id: str,
    output_dataset_id: str,
    model_id: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    num_examples: Optional[int] = None,
    text_column_name: Optional[str] = None,
):
    # Run the pipeline
    pipeline = create_pipeline(
        model_id=model_id,
        endpoint_name=endpoint_name,
        text_column_name=text_column_name,
    )
    # Start building the parameters dictionary
    parameters = {
        "load_dataset": {
            "repo_id": dataset_id,
        },
        "paraphrase_text": {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": 1000,
                    "temperature": 0.7,
                    "do_sample": True,
                    "grammar": {"type": "json", "value": schema},
                },
            },
        },
    }

    # Conditionally add num_examples if it's provided
    if num_examples is not None:
        parameters["load_dataset"]["num_examples"] = num_examples

    # Run the pipeline with the dynamically built parameters
    distiset = pipeline.run(parameters, use_cache=True)

    # Push to the hub
    distiset.push_to_hub(output_dataset_id)
    update_card(output_dataset_id)


if __name__ == "__main__":
    run_pipeline(
        INPUT_DATASET_ID,
        output_dataset_id=OUTPUT_DATASET_ID,
        endpoint_name=END_POINT_NAME,
        num_examples=NUM_EXAMPLES,
        text_column_name=TEXT_COLUMN_NAME,
    )
