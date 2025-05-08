import pytest
from panda_guard.llms import LLMGenerateConfig
from panda_guard.llms.oai import (
    OpenAiChatLLMConfig,
    OpenAiLLMConfig,
    OpenAiLLM,
    OpenAiChatLLM,
)
from panda_guard.llms.hf import HuggingFaceLLMConfig, HuggingFaceLLM
from panda_guard.llms.vllm import VLLMLLMConfig, VLLMLLM


def check_response(msg):
    return len(msg[-1]["content"]) > 0


@pytest.fixture(scope="function")
def input_prompt():
    prompt = "Hello, how are you today?"
    return prompt


@pytest.fixture(scope="function")
def llm_gen_config():
    # use fixture factory
    def _llm_gen_config():
        return LLMGenerateConfig(
            max_n_tokens=128, temperature=1.0, logprobs=False, seed=42
        )

    return _llm_gen_config


def test_openaillm_gen(input_prompt, llm_gen_config):
    config = OpenAiLLMConfig(
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        base_url="http://172.18.129.80:8000/v1",
    )
    llm = OpenAiLLM(config=config)
    messages = [{"role": "user", "content": input_prompt}]
    response_msg = llm.generate(messages=messages, config=llm_gen_config())
    assert check_response(response_msg) is True


def test_openaichatllm_gen(input_prompt, llm_gen_config):
    config = OpenAiChatLLMConfig(
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        base_url="http://172.18.129.80:8000/v1",
    )
    llm = OpenAiChatLLM(config=config)
    messages = [{"role": "user", "content": input_prompt}]
    response_msg = llm.generate(messages=messages, config=llm_gen_config())
    assert check_response(response_msg) is True


def test_hfllm_gen(input_prompt, llm_gen_config):
    config = HuggingFaceLLMConfig(
        model_name="Qwen/Qwen2-7B-Instruct",
        device_map="cuda:0",
    )
    llm = HuggingFaceLLM(config)
    messages = [{"role": "user", "content": input_prompt}]
    response_msg = llm.generate(messages=messages, config=llm_gen_config())
    assert check_response(response_msg) is True


def test_vllmllm_gen(input_prompt, llm_gen_config):
    config = VLLMLLMConfig(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        tensor_parallel_size=1,  # Use 1 GPU
        gpu_memory_utilization=0.8,
    )
    llm = VLLMLLM(config)
    messages = [{"role": "user", "content": input_prompt}]
    response_msg = llm.generate(messages=messages, config=llm_gen_config())
    assert check_response(response_msg) is True
