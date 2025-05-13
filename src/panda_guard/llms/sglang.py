import requests
from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Tuple, Generator
from panda_guard.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig
from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd


@dataclass
class SGLangLLMConifg(BaseLLMConfig):
    llm_type: str = field(default="SGLangLLM")
    model_name: [str, Any] = field(default="qwen/qwen2.5-0.5b-instruct")


class SGLangLLM(BaseLLM):
    def __init__(self, config: SGLangLLMConifg):
        super().__init__(config)
        self.model_name = config.model_name
        server_process, port = launch_server_cmd(
            f"""
        Python3 -m sglang.launch_server --model-path {self.model_name} \
         --host 0.0.0.0
        """
        )
        self.server_process, self.port = server_process, port
        self.url = f"http://localhost:{self.port}/v1/chat/completions"
        wait_for_server(f"http://localhost:{port}")

    def generate(self, messages: List[Dict[str, str]], config: LLMGenerateConfig):
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_n_tokens": config.max_n_tokens,
            "temperature": config.temperature,
        }
        response = requests.post(self.url, json=data)
        print_highlight(response.json())

    def batch_generate(self, batch_messages, config):
        return super().batch_generate(batch_messages, config)

    def continual_generate(self, messages, config):
        raise NotImplementedError

    def evaluate_log_likelihood(self, messages, config, require_grad=False):
        raise NotImplementedError


if __name__ == "__main__":
    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=100, temperature=0.7, logprobs=True, seed=42
    )
    llm = SGLangLLM(config=SGLangLLMConifg())
    msg = [{"role": "user", "content": "What is the capital of France?"}]
    llm.generate(messages=msg, config=llm_gen_config)
