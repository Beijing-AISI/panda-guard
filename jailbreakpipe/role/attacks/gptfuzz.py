from typing import Dict, List
from dataclasses import dataclass, field
from jailbreakpipe.role.attacks import BaseAttacker, BaseAttackerConfig
from jailbreakpipe.role.attacks.attacker_registry import register_attacker
from jailbreakpipe.utils import is_user_turn
from jailbreakpipe.llms import (
    create_llm,
    BaseLLMConfig,
    LLMGenerateConfig,
    HuggingFaceLLMConfig,
    HuggingFaceLLM,
)

import pandas as pd
import os
import sys

from .gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from .gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy,
    OpenAIMutatorCrossOver,
    OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar,
    OpenAIMutatorRephrase,
    OpenAIMutatorShorten,
)
from .gptfuzzer.fuzzer import GPTFuzzer
from .gptfuzzer.utils.predict import RoBERTaPredictor


@dataclass
class GPTFuzzAttackerConfig(BaseAttackerConfig):
    """
    Configuration for the GPTFuzz Attacker.

    :param attacker_cls: Class of the attacker, default is "TransferAttacker".  攻击者的类型，默认值为 "GPTFuzzAttacker"
    :param attacker_name: Name of the attacker.  攻击者的名称
    :param attacker_llm_config: Configuration of attacker llm. 攻击时使用的llm配置
    :param attacker_llm_gen_config: Generation configuration for the attacker's LLM. 攻击者 LLM 的生成配置
    :param target_llm_config: Configuration of target llm. 攻击目标的llm配置
    :param target_llm_gen_config: Generation configuration for the attacker's LLM. 被攻击的LLM 的生成配置
    :param initial_seed: initial seed. 有害prompt template集合, 当作初始种子
    :param predict_model: A model for determining whether a jailbreak attack has succeeded, with an output of 0 or 1. 用于判断是否越狱攻击成功的模型, 输出为0或者1
    """

    attacker_cls: str = field(default="GPTFuzzAttacker")
    attacker_name: str = field(default=None)
    attacker_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    attacker_llm_gen_config: LLMGenerateConfig = field(default=None)
    target_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    target_llm_gen_config: LLMGenerateConfig = field(default=None)
    initial_seed: list = field(default=None)
    predict_model: str = field(default=None)


@register_attacker
class GPTFuzzAttacker(BaseAttacker):
    """
    GPTFuzz Attacker Implementation that substitutes the user message with a pre-formulated attack prompt.

    :param config: Configuration for the GPTFuzzAttacker.  用于GPTFuzzAttacker的配置
    """

    def __init__(self, config: GPTFuzzAttackerConfig):
        super().__init__(config)

        self.config = config

        self.attack_llm = create_llm(config=config.attacker_llm_config)
        self.target_llm = create_llm(config=config.target_llm_config)

        self.initial_seed = pd.read_csv(config.initial_seed)["text"].tolist()
        self.predict_model = RoBERTaPredictor(config.predict_model)

    def attack(self, messages: List[Dict[str, str]], **kwargs) -> List[Dict[str, str]]:
        """
        Execute an attack by transferring a reformulated request into the conversation.

        :param messages: List of messages in the conversation.  对话中的消息列表
        :param kwargs: Additional parameters for the attack, must include "request_reformulated".  额外攻击参数
        :return: Prompts containing harmful attacks on the target, is of the form “role: user, content: xx”. 含有目标的有害攻击的prompt, 是"role: user, content: xx的形式"
        """

        question = messages[0]["content"]

        fuzzer = GPTFuzzer(
            question=question,
            # target_model=openai_model,
            target=self.target_llm,
            predictor=self.predict_model,
            initial_seed=self.initial_seed,
            mutate_policy=MutateRandomSinglePolicy(
                [
                    OpenAIMutatorCrossOver(
                        self.attack_llm, self.config.attacker_llm_gen_config
                    ),
                    OpenAIMutatorExpand(
                        self.attack_llm, self.config.attacker_llm_gen_config
                    ),
                    OpenAIMutatorGenerateSimilar(
                        self.attack_llm, self.config.attacker_llm_gen_config
                    ),
                    OpenAIMutatorRephrase(
                        self.attack_llm, self.config.attacker_llm_gen_config
                    ),
                    OpenAIMutatorShorten(
                        self.attack_llm, self.config.attacker_llm_gen_config
                    ),
                ],
                concatentate=True,
            ),
            select_policy=MCTSExploreSelectPolicy(),
            energy=1,
            max_jailbreak=1,
            max_query=200,
            generate_in_batch=False,
        )

        messages = fuzzer.run(self.config.target_llm_gen_config)

        return messages
