# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 22:00
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : base.py
# explain   :

import abc
from typing import Dict, List, Union, Any, Tuple
from dataclasses import dataclass, field
import concurrent.futures


@dataclass
class LLMGenerateConfig:
    """
    Configuration for LLM generation.

    :param max_n_tokens: Maximum number of tokens to generate.  最大生成的词汇数量
    :param temperature: Temperature for sampling randomness.  用于样本随机的温度参数
    :param logprobs: Whether to return log probabilities.  是否返回logprobs
    :param seed: Seed for reproducibility.  用于可重复的种子
    """

    max_n_tokens: int = field(default=None)
    temperature: float = field(default=None)
    logprobs: bool = field(default=False)
    seed: int = field(default=None)


@dataclass
class BaseLLMConfig(abc.ABC):
    """
    Base configuration for LLM.

    :param llm_type: Type of the LLM.  LLM的类型
    :param model_name: Name of the model.  模型的名称
    """

    llm_type: str = field(default=None)
    model_name: str = field(default=None)


class BaseLLM(abc.ABC):
    """
    Abstract base class for LLM.

    :param config: Configuration object for LLM.  LLM的配置对象
    """

    def __init__(self, config: BaseLLMConfig):
        self._CLS = config.llm_type
        self._NAME = config.model_name

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_requires = 0

    @abc.abstractmethod
    def generate(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]:
        """
        Abstract method for generating response from LLM.

        :param messages: List of messages for input.  输入的消息列表
        :param config: Configuration for generation.  生成配置
        :return: Generated response or responses with log probabilities.  返回生成的应答或启用百分比的应答
        """
        pass

    @abc.abstractmethod
    def evaluate_log_likelihood(
        self,
        messages: List[Dict[str, str]],
        config: LLMGenerateConfig,
        require_grad=False,
    ) -> List[float]:
        """
        Abstract method for evaluating log likelihood of messages.

        :param messages: List of messages to evaluate.  需要评估的消息列表
        :param config: Configuration for generation.  生成配置
        :param require_grad: Determine whether returned logprobs has grad
        :return: List of log likelihoods.  返回的百分比列表
        """
        pass

    @abc.abstractmethod
    def continual_generate(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]:
        """
        Remove EOS token in formatted prompt. Manually add generation prompt.

        :param messages: List of messages for input.  输入的消息列表
        :param config: Configuration for generation.  生成配置
        :return: Generated response or responses with log probabilities.  返回生成的应答或启用百分比的应答
        """
        pass

    def batch_generate(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: LLMGenerateConfig,
    ) -> List[Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]]:
        """
        Generate responses for a batch of messages concurrently.

        :param batch_messages: List of batches of messages.  消息的批量列表
        :param config: Configuration for generation.  生成配置
        :return: List of generated responses.  返回生成的应答列表
        """
        configs = [config] * len(batch_messages)
        if config.seed is not None:
            seeds = list(range(config.seed, config.seed + len(batch_messages)))
            for i, c in enumerate(configs):
                c.seed = seeds[i]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.generate, messages, config)
                for config, messages in zip(configs, batch_messages)
            ]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
        return results

    def reset(self):
        """
        Reset the token counts and the number of requests.

        重置词汇计数和请求次数
        """
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_requires = 0

    def update(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        num_requires: int,
    ):
        """
        Update the token counts and number of requests.

        :param prompt_tokens: Number of tokens in prompt.  提示中的词汇数
        :param completion_tokens: Number of tokens in completion.  完成中的词汇数
        :param num_requires: Number of requests made.  请求次数
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.num_requires += num_requires

    @property
    def total_tokens(self):
        """
        Get the total number of tokens used.

        :return: Total number of tokens.  返回的词汇总数
        """
        return self.prompt_tokens + self.completion_tokens

    @property
    def avg_tokens(self):
        """
        Get the average number of tokens per request.

        :return: Average number of tokens.  返回每次请求的词汇平均值
        """
        return self.total_tokens / self.num_requires
