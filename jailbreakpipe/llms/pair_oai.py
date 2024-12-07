import abc
import os
from typing import Dict, List, Union, Any, Tuple
from dataclasses import dataclass, field
import concurrent.futures
import torch
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

from jailbreakpipe.llms.llm_registry import register_llm
from jailbreakpipe.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig


@dataclass
class PairAttackerOAiLLMConfig(BaseLLMConfig):
    """
    OpenAI LLM Configuration for PAIR

    :param llm_type: Type of LLM, default is "OpenAiLLM".  LLM的类型，默认值为 "OpenAiLLM"
    :param model_name: Name of the model.  模型的名称
    :param base_url: Base URL for the OpenAI API.  OpenAI API的基础URL
    :param api_key: API key for accessing OpenAI.  访问OpenAI的API密钥
    """

    llm_type: str = field(default="OpenAiLLM")
    model_name: str = field(default=None)
    base_url: str = field(default=None)
    api_key: str = field(default="KEY HERE")


@register_llm
class PairAttackerOAiLLM(BaseLLM):
    """
    OpenAI LLM Implementation. For PAIR Attacker

    :param config: Configuration for OpenAI LLM.  用于OpenAI LLM的配置
    """

    def __init__(self, config: PairAttackerOAiLLMConfig):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=True,
        )
        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def generate(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]:
        """
        Generate a response for a given input using OpenAI API.

        :param messages: List of input messages.  输入的消息列表
        :param config: Configuration for LLM generation.  生成配置
        :return: Generated response or response with logprobs.  返回生成的应答或启用logprobs的应答
        """
        if "gemma" in self._NAME.lower() and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
            messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, continual_final_message=True
        )

        tokens = self.tokenizer(prompt, return_tensors="pt")

        if tokens["input_ids"].shape[1] > 3840:
            truncated_tokens = tokens["input_ids"][:, :3840]
            prompt = self.tokenizer.decode(
                truncated_tokens[0], skip_special_tokens=True
            )

        response = self.client.completions.create(
            model=self._NAME,
            prompt=prompt,
            max_tokens=config.max_n_tokens,
            temperature=config.temperature,
            logprobs=config.logprobs,
        )
        content = response.choices[0].text
        # messages.append({"role": "assistant", "content": content})
        messages[-1]["content"] += content

        self.update(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            1,
        )

        if config.logprobs:
            logs = response.choices[0].logprobs.token_logprobs
            return messages, logs

        return messages

    def evaluate_log_likelihood(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> List[float]:
        """
        Evaluate the log likelihood of the given messages.

        :param messages: List of messages for evaluation.  需要评估的消息列表
        :param config: Configuration for LLM generation.  生成配置
        :return: List of log likelihood values.  返回的log likelihood值列表
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, continual_final_message=True
        )
        response = self.client.completions.create(
            model=self._NAME,
            prompt=prompt,
            max_tokens=0,
            temperature=config.temperature,
            logprobs=1,
            echo=True,
        )
        logprobs = response.choices[0].logprobs.token_logprobs

        self.update(response.usage.prompt_tokens, 0, 1)

        return logprobs[-len(self.tokenizer(messages[-1]["content"]).input_ids) :]


if __name__ == "__main__":
    from jailbreakpipe.llms import LLMS

    print(LLMS)

    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=128, temperature=1.0, logprobs=True, seed=42
    )

    config = PairAttackerOAiLLMConfig(
        model_name="meta-llama/Meta-Llama-3.1-70B-Instruct",
        base_url="http://172.18.129.80:8000/v1",
    )
    llm = PairAttackerOAiLLM(config)

    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am"},
    ]

    results = llm.generate(messages=messages, config=llm_gen_config)
    print(results, len(results))
