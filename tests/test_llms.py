import os
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
from panda_guard.llms.gemini import GeminiLLMConfig, GeminiLLM
from panda_guard.llms.claude import ClaudeLLM, ClaudeLLMConfig


def check_response(msg):
    return len(msg[-1]["content"]) > 0


@pytest.fixture(scope="function")
def input_prompt():
    prompt = "Hello, how are you today?"
    return prompt


class TestLLMs:
    def llm_gen_config(self):
        # use fixture factory
        return LLMGenerateConfig(
            max_n_tokens=128, temperature=1.0, logprobs=False, seed=42
        )

    @pytest.mark.api
    def test_gemini_gen(self, input_prompt):
        config = GeminiLLMConfig(
            model_name="gemini-1.5-pro", api_key=os.getenv("GOOGLE_API_KEY")
        )

        llm = GeminiLLM(config)
        messages = [{"role": "user", "content": input_prompt}]
        response_msg = llm.generate(messages=messages, config=self.llm_gen_config())
        assert check_response(response_msg) is True

    @pytest.mark.api
    def test_claude_gen(self, input_prompt):
        config = ClaudeLLMConfig(
            model_name="claude-3-opus-20240229", api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        llm = ClaudeLLM(config)
        messages = [{"role": "user", "content": input_prompt}]
        response_msg = llm.generate(messages=messages, config=self.llm_gen_config())
        assert check_response(response_msg) is True

    @pytest.mark.api
    def test_openaillm_gen(self, input_prompt):
        config = OpenAiLLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-YOUR_ACTUAL_API_KEY_HERE",
        )
        llm = OpenAiLLM(config=config)
        messages = [{"role": "user", "content": input_prompt}]
        response_msg = llm.generate(messages=messages, config=self.llm_gen_config())
        assert check_response(response_msg) is True

    @pytest.mark.api
    def test_openaichatllm_gen(self, input_prompt):
        config = OpenAiChatLLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-YOUR_ACTUAL_API_KEY_HERE",
        )
        llm = OpenAiChatLLM(config=config)
        messages = [{"role": "user", "content": input_prompt}]
        response_msg = llm.generate(messages=messages, config=self.llm_gen_config())
        assert check_response(response_msg) is True

    @pytest.mark.skip(reason="skip hf llm")
    def test_hfllm_gen(self, input_prompt):
        config = HuggingFaceLLMConfig(
            model_name="Qwen/Qwen3-0.6B",
            device_map="sequential",
        )
        llm = HuggingFaceLLM(config)
        messages = [{"role": "user", "content": input_prompt}]
        response_msg = llm.generate(messages=messages, config=self.llm_gen_config())
        assert check_response(response_msg) is True

    @pytest.mark.skip(reason="skip vllm llm")
    def test_vllmllm_gen(self, input_prompt):
        config = VLLMLLMConfig(
            model_name="Qwen/Qwen3-0.6B",
            tensor_parallel_size=1,  # Use 1 GPU
            gpu_memory_utilization=0.8,
        )
        llm = VLLMLLM(config)
        messages = [{"role": "user", "content": input_prompt}]
        response_msg = llm.generate(messages=messages, config=self.llm_gen_config())
        assert check_response(response_msg) is True
