# Synthetic dataset generation techniques: Self-Instruct

_note_ also published at: https://huggingface.co/blog/davanstrien/self-instruct.

For training an LLM to be better at following instructions or functioning as a chat model, you usually want a dataset with some combination of instructions and responses. Since creating this data by hand can be very time-consuming, more people are using LLMs to generate this data.

In its simplest form, you could create synthetic instruction following datasets using an LLM to generate responses to handwritten prompts/instructions. However, for many applications, there are a considerable number of prompts you may want to have in your final datasets. Creating all this data by hand will be challenging while ensuring its diversity. There are various ways in which you can try to remove this bottleneck.

In this blog post, I'll discuss the technique outlined in the paper [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://huggingface.co/papers/2212.10560), which, as the paper's title suggests, aims to overcome the need to generate instructions by hand.

> Self-Instruct is a framework that helps language models improve their ability to follow natural language instructions. It does this by using the model's own generations to create a large collection of instructional data. With Self-Instruct, it is possible to improve the instruction-following capabilities of language models without relying on extensive manual annotation. [source](https://github.com/yizhongw/self-instruct?tab=readme-ov-file#introduction)

The paper outlines a pipeline that bootstraps from an initial seed dataset of instructions to a larger dataset of synthetically generated instructions.
![Pipeline Diagram](https://github.com/yizhongw/self-instruct/raw/main/docs/pipeline.JPG)

The authors include steps for generating instructions and a filtering step to clean up the data in the paper. Since our goal is to focus on the core technique of a particular paper, we'll focus only on the instruction generation part. This step can also be combined with other approaches to data filtering that have been introduced since this paper was published (or your own custom filters).

## Instruction Generation

Returning to our original challenge: how do we generate instructions without writing them all by hand? As you can see in the diagram above, the steps involve sampling from the original seeds, filtering the seed tasks to see if they are classification tasks or not, and then generating the new instructions. After the new instructions are generated, they are filtered and added to the task pool. In this way, you can keep creating new instructions from your initial seed tasks and growing the seed task pool. Using the data filtering steps aims to ensure you still have diversity in the prompts and avoid adding very repetitive instructions to your dataset.

### What does this look like in practice?

Let's take a look at an example from the 175 initial seeds task dataset:

```
{"id": "seed_task_0",
"name": "breakfast_suggestion",
"instruction": "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?",
"instances": [{"input": "", "output": "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup watter, totalling about 550 calories. The 4 strips of bacon contains about 200 calories."}],
"is_classification": false}
```

As you can see, this row contains fairly standard instructions, such as "Is there anything...", some responses (i.e., the instances field), and a label indicating if it's a classification task. The paper outlines two main approaches to generating new instructions from this data. If an instruction is a classification task, one prompting approach is used; if it's a standard generation task, another prompt is used. Let's start with how the extra nonclassification task instructions prompt looks:

```
> Come up with a series of tasks:
>
> Task 1: {instruction for existing task 1}
> Task 2: {instruction for existing task 2}
> Task 3: {instruction for existing task 3}
> Task 4: {instruction for existing task 4}
> Task 5: {instruction for existing task 5}
> Task 6: {instruction for existing task 6}
> Task 7: {instruction for existing task 7}
> Task 8: {instruction for existing task 8}
> Task 9:
```

As you can see, the prompt gives the llm some examples of tasks and encourages the model to generate new instructions. A crucial detail to note is in the original paper; the authors used GPT3 rather than an instruction-tuned/chat model. Since this is not an instruction-tuned model, a prompt that gives a few examples in a structured format can often be better for guiding the model toward a useful set of generations.

We can see what this process looks like in practice (using `huggingface_hub` and the BigScience Bloom model in place of the GPT-3):

```python
from huggingface_hub import InferenceClient

client = InferenceClient('bigscience/bloom')

def encode_prompt(prompt_instructions, classification=False):
    """Encode multiple prompt instructions into a single string."""
    if classification:
        prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
    else:
        prompt = "Come up with a series of tasks:\n"
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt

prompt = encode_prompt(dataset['instruction']) #
```

For a non-classification task, this produces a prompt which looks like this:

```
Come up with a series of tasks:
1. Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?
2. What is the relation between the given pairs?
3. Generate a one-sentence description for each of the following people.
4. Describe a situation in which the given stereotype can harm you.
5. Generate an appropriate subjective title for the following email
6. How do you answer this question in a job interview?
7. Brainstorm a list of possible New Year's resolutions.
8. Explain the following idiom to me, and try to give me some examples.
9.
```

We can then pass this prompt to an LLM.

```python
client.text_generation(prompt, return_full_text=False, temperature=0.7, max_new_tokens=512)
>>>  Think of a time when you were incredibly confident, and explain why.\n10. What is the difference between a real and normal friend?
```

We can see the LLM responds with new instructions (we also get some extra text from the LLM). If we wanted to use this in practice, we could do more work to optimize the generation parameters (temperature, etc.).

The process and prompts for the text classification tasks are slightly different. To try to avoid the LLM just responding with the label token, they put the label first and then show the text that generated that label i.e. something like this:

```
Instruction: Find out if the given text is positive about the company discussed.
Class Label: Positive
Input: Hugging Face is a wonderful platform for machine learning developers.
```

## Using Self Instruct

This paper has had a very big impact both in academic research (over 1,000 [citations](https://www.semanticscholar.org/paper/Self-Instruct%3A-Aligning-Language-Models-with-Wang-Kordi/e65b346d442e9962a4276dc1c1af2956d9d5f1eb#citing-papers) and in the practical adoption of the method by the community (you can find some datasets citing the method [here](https://huggingface.co/datasets?other=arxiv:2212.10560).

There are several implementations of the Self Instruct method:

- Official GitHub repository: [https://github.com/yizhongw/self-instruct](https://github.com/yizhongw/self-instruct)
- [Distilabel implementation](https://distilabel.argilla.io/latest/reference/distilabel/steps/tasks/#distilabel.steps.tasks.SelfInstruct)
- [airoboros](https://github.com/jondurbin/airoboros): a modified version of self instruct.

In practice, most uses of this approach have moved away from strictly following the prompts/approach outlined in the paper. Since the quality of open and closed-source instruction following models has significantly improved since his paper was published, it often makes more sense to use this to prompt a model to generate new instructions more directly.

While the exact approach outlined in the paper is often adapted, the paper is still helpful for better understanding how to approach synthetic data generation.
