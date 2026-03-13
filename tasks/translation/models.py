import os

import torch
import concurrent.futures
import tqdm
from vllm import LLM, SamplingParams
import openai


def build_model(
    model_name: str,
    infer_backend: str,
    tokenizer: str = None,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = False,
    base_url: str = None,
    num_concurrent: int = 1,
    **kwargs
):
    match infer_backend:
        case "vllm":
            return VLLMCausalLLM(
                model_path=model_name,
                tokenizer=tokenizer,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
        case "openai_api":
            api_key = os.getenv("OPENAI_API_KEY", None)
            if api_key is None:
                raise ValueError(
                    "API key not found. Please set the `OPENAI_API_KEY` environment variable."
                )
            return OpenAILLM(model_name=model_name, base_url=base_url, api_key=api_key, num_concurrent=num_concurrent, **kwargs)
        case _:
            raise ValueError(f"Unknown infer_backend: {infer_backend}")


class VLLMCausalLLM:
    def __init__(self, model_path: str, tokenizer: str = None, tensor_parallel_size: int = 1, trust_remote_code: bool = False, **kwargs):
        if tensor_parallel_size is None or tensor_parallel_size < 0:
            tensor_parallel_size = torch.cuda.device_count()
        self.model = LLM(
            model=model_path,
            tokenizer=tokenizer,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=trust_remote_code,
            **kwargs
        )

    def chat(self, messages, temperature=0.0, max_tokens=512, chat_template_kwargs=None):
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        kwargs = {}
        if chat_template_kwargs:
            kwargs["chat_template_kwargs"] = chat_template_kwargs
        response = self.model.chat(messages, sampling_params=sampling_params, **kwargs)
        outputs = [r.outputs[0].text.strip() for r in response]
        return outputs


class OpenAILLM:
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None, num_concurrent: int = 1, **kwargs):
        if api_key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, **kwargs)
        self.model_name = model_name
        self.num_concurrent = num_concurrent

    def chat(self, messages, temperature=0.0, max_tokens=512):
        if self.num_concurrent > 1:
            outputs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_concurrent) as executor:
                futures = [
                    executor.submit(
                        self._chat,
                        message,
                        temperature,
                        max_tokens,
                    )
                    for message in messages
                ]
                for future in tqdm.tqdm(futures, total=len(messages)):
                    outputs.append(future.result())
        else:
            outputs = [
                self._chat(message, temperature, max_tokens)
                for message in tqdm.tqdm(messages)
            ]
        return outputs


    def _chat(self, message, temperature, max_tokens):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        output = response.choices[0].message.content.strip()
        return output