from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Tuple, Generator
import ollama
from ollama import chat
from ollama import ChatResponse
from panda_guard.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig
import subprocess
import atexit


@dataclass
class OllamaLLMConfig(BaseLLMConfig):
    llm_type: str = field(default="OllamaLLM")
    model_name: [str, Any] = field(default="qwen3:0.6b")


class OllamaLLM(BaseLLM):
    def __init__(self, config: OllamaLLMConfig):
        super().__init__(config)
        self.model_name = config.model_name
        ollama.pull(self.model_name)

        # auto tear down
        atexit.register(self.ollama_teardown)

    def generate(self, messages: List[Dict[str, str]], config: LLMGenerateConfig):
        response: ChatResponse = chat(
            model=self.model_name,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.message.content})
        return messages

    def ollama_teardown(self):
        print("Stopping Ollama session...")
        command = ["ollama", "stop", self.model_name]
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # If check=True was used and no CalledProcessError was raised, the command succeeded.
        print("\nCommand executed successfully!")
        print(f"Return Code: {result.returncode}")

    def continual_generate(self, messages, config):
        raise NotImplementedError

    def evaluate_log_likelihood(self, messages, config, require_grad=False):
        raise NotImplementedError

    def batch_generate(self, batch_messages, config):
        return super().batch_generate(batch_messages, config)


if __name__ == "__main__":
    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=100, temperature=0.7, logprobs=True, seed=42
    )
    llm = OllamaLLM(config=OllamaLLMConfig())
    msg = [{"role": "user", "content": "Why is the sky red?"}]
    message = llm.generate(messages=msg, config=llm_gen_config)
    print(message)
    # llm.ollama_teardown()
