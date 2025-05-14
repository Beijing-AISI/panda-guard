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
from panda_guard.llms.vllm_llm import VLLMLLMConfig, VLLMLLM
from panda_guard.llms.gemini import GeminiLLMConfig, GeminiLLM
from panda_guard.llms.claude import ClaudeLLM, ClaudeLLMConfig
from panda_guard.llms.sglang_llm import SGLangLLM, SGLangLLMConifg
from panda_guard.llms.ollama_llm import OllamaLLM, OllamaLLMConfig


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
    @pytest.mark.skipif(
        not os.getenv(
            "GEMINI_API_KEY"
        ),  # Condition: True if GEMINI_API_KEY is not set or is empty
        reason="GEMINI_API_KEY environment variable not set or is empty",
    )
    def test_gemini_gen(self, input_prompt):
        config = GeminiLLMConfig(
            model_name="gemini-1.5-pro", api_key=os.getenv("GEMINI_API_KEY")
        )

        llm = GeminiLLM(config)
        messages = [{"role": "user", "content": input_prompt}]
        response_msg = llm.generate(messages=messages, config=self.llm_gen_config())
        assert check_response(response_msg) is True

    @pytest.mark.api
    @pytest.mark.skipif(
        not os.getenv(
            "ANTHROPIC_API_KEY"
        ),  # Condition: True if ANTHROPIC_API_KEY is not set or is empty
        reason="ANTHROPIC_API_KEY environment variable not set or is empty",
    )
    def test_claude_gen(self, input_prompt):
        config = ClaudeLLMConfig(
            model_name="claude-3-opus-20240229", api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        llm = ClaudeLLM(config)
        messages = [{"role": "user", "content": input_prompt}]
        response_msg = llm.generate(messages=messages, config=self.llm_gen_config())
        assert check_response(response_msg) is True

    @pytest.mark.api
    @pytest.mark.skipif(
        not os.getenv(
            "OPENAI_API_KEY"
        ),  # Condition: True if OPENAI_API_KEY is not set or is empty
        reason="OPENAI_API_KEY environment variable not set or is empty",
    )
    def test_openaillm_gen(self, input_prompt):
        config = OpenAiLLMConfig(
            model_name="gpt-4.1",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        llm = OpenAiLLM(config=config)
        messages = [{"role": "user", "content": input_prompt}]
        response_msg = llm.generate(messages=messages, config=self.llm_gen_config())
        assert check_response(response_msg) is True

    @pytest.mark.api
    @pytest.mark.skipif(
        not os.getenv(
            "OPENAI_API_KEY"
        ),  # Condition: True if OPENAI_API_KEY is not set or is empty
        reason="OPENAI_API_KEY environment variable not set or is empty",
    )
    def test_openaichatllm_gen(self, input_prompt):
        config = OpenAiChatLLMConfig(
            model_name="gpt-4.1",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        llm = OpenAiChatLLM(config=config)
        messages = [{"role": "user", "content": input_prompt}]
        response_msg = llm.generate(messages=messages, config=self.llm_gen_config())
        assert check_response(response_msg) is True

    # @pytest.mark.skip(reason="skip hf llm")
    def test_hfllm_gen(self, input_prompt):
        config = HuggingFaceLLMConfig(
            model_name="Qwen/Qwen3-0.6B",
            device_map="sequential",
        )
        llm = HuggingFaceLLM(config)
        messages = [{"role": "user", "content": input_prompt}]
        response_msg = llm.generate(messages=messages, config=self.llm_gen_config())
        assert check_response(response_msg) is True

    # @pytest.mark.skip(reason="skip vllm llm")
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

    # @pytest.mark.skip(reason="skip sglang llm")
    def test_sglang_gen(self, input_prompt):
        llm_gen_config = LLMGenerateConfig(
            max_n_tokens=100, temperature=0.7, logprobs=True, seed=42
        )
        llm = SGLangLLM(config=SGLangLLMConifg(model_name="qwen/qwen2.5-0.5b-instruct"))

        msg = [{"role": "user", "content": input_prompt}]
        msg = llm.generate(messages=msg, config=llm_gen_config)
        assert check_response(msg) is True

    # @pytest.mark.skip(reason="skip ollama llm")
    def test_ollama_gen(self, input_prompt):
        llm_gen_config = LLMGenerateConfig(
            max_n_tokens=100, temperature=0.7, logprobs=True, seed=42
        )
        llm = OllamaLLM(config=OllamaLLMConfig(model_name="qwen3:0.6b"))
        msg = [{"role": "user", "content": input_prompt}]
        message = llm.generate(messages=msg, config=llm_gen_config)
        assert check_response(message) is True
