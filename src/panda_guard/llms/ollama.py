from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Tuple, Generator
import ollama
from ollama import chat
from ollama import ChatResponse
from panda_guard.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig


@dataclass
class OllamaLLMConfig(BaseLLMConfig):
    llm_type: str = field(default="OllamaLLM")
    model_name: [str, Any] = field(default="qwen3:0.6b")


class OllamaLLM(BaseLLM):
    def __init__(self, config: OllamaLLMConfig):
        super().__init__(config)
        self.model_name = config.model_name
        ollama.pull(self.model_name)

    def generate(self, messages: List[Dict[str, str]], config: LLMGenerateConfig):
        response: ChatResponse = chat(
            model=self.model_name,
            messages=messages,
        )
        print(response["message"]["content"])
        # or access fields directly from the response object
        print(response.message.content)


if __name__ == "__main__":
    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=100, temperature=0.7, logprobs=True, seed=42
    )
    llm = OllamaLLM(config=OllamaLLMConfig())
    msg = [{"role": "user", "content": "Why is the sky blue?"}]
    llm.generate(messages=msg, config=llm_gen_config)
