import random
import warnings
from typing import Any, Dict, List, Optional, Union

from distilabel.llms import InferenceEndpointsLLM
from distilabel.llms.typing import GenerateOutput
from distilabel.steps.tasks.typing import ChatType
from pydantic import (
    validate_call,
)


class InferenceEndpointsLLMWithGrammar(InferenceEndpointsLLM):
    @validate_call
    async def agenerate(  # type: ignore
        self,
        input: ChatType,
        max_new_tokens: int = 128,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: Optional[float] = None,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        stop_sequences: Optional[Union[str, List[str]]] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        watermark: bool = False,
        grammar: Optional[Dict[str, Any]] = None,
    ) -> "GenerateOutput":
        """Generates completions for the given input using the OpenAI async client.

        Args:
            input: a single input in chat format to generate responses for.
            max_new_tokens: the maximum number of new tokens that the model will generate.
                Defaults to `128`.
            frequency_penalty: the repetition penalty to use for the generation. Defaults
                to `0.0`. Only applies if `use_openai_client=True`.
            presence_penalty: the presence penalty to use for the generation. Defaults to
                `0.0`. Only applies if `use_openai_client=True`.
            repetition_penalty: the repetition penalty to use for the generation. Defaults
                to `None`. Only applies if `use_openai_client=False`.
            temperature: the temperature to use for the generation. Defaults to `1.0`.
            do_sample: whether to use sampling for the generation. Defaults to `False`.
                Only applies if `use_openai_client=False`.
            top_k: the top-k value to use for the generation. Defaults to `0.8`, since neither
                `0.0` nor `1.0` are valid values in TGI.
            top_p: the top-p value to use for the generation. Defaults to `1.0`.
            typical_p: the typical-p value to use for the generation. Defaults to `0.5`.
            stop_sequences: either a single string or a list of strings containing the sequences
                to stop the generation at. Defaults to `None`, but will be set to the
                `tokenizer.eos_token` if available.
            return_full_text: whether to return the full text of the completion or just the
                generated text. Defaults to `False`, meaning that only the generated text will be
                returned.
            seed: the seed to use for the generation. Defaults to `None`.
            watermark: whether to add the watermark to the generated text. Defaults to `None`.

        Returns:
            A list of lists of strings containing the generated responses for each input.
        """
        if stop_sequences is not None:
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            if len(stop_sequences) > 4:
                warnings.warn(
                    "Only up to 4 stop sequences are allowed, so keeping the first 4 items only.",
                    UserWarning,
                    stacklevel=2,
                )
                stop_sequences = stop_sequences[:4]
        if self.use_openai_client and grammar is not None:
            raise ValueError(
                "Grammar is not supported when using the OpenAI client. Please set `use_openai_client=False`."
            )
        if self.use_openai_client:
            return await self._openai_agenerate(
                input=input,
                max_new_tokens=max_new_tokens,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences,
            )
        if self._tokenizer is not None:
            prompt = self._tokenizer.apply_chat_template(  # type: ignore
                conversation=input,  # type: ignore
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # TODO: should we apply a default chat template here instead? e.g. ChatML
            prompt = "\n".join([message["content"] for message in input])

        try:
            completion = await self._aclient.text_generation(  # type: ignore
                prompt=prompt,  # type: ignore
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences,
                return_full_text=return_full_text,
                watermark=watermark,
                grammar=grammar,  # Pass grammar to text_generation
                seed=seed or random.randint(0, 2147483647),
            )
            return [completion]
        except Exception as e:
            self._logger.warning(  # type: ignore
                f"⚠️ Received no response using Inference Client (model: '{self.model_name}')."
                f" Finish reason was: {e}"
            )
            return [None]
