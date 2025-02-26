# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/4 14:37
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : inference.py
# explain   : Inference pipeline for executing attack and defense phases.

import abc
from typing import Dict, List, Union, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from jailbreakpipe.llms import BaseLLM, BaseLLMConfig
from jailbreakpipe.role import (
    create_attacker,
    create_defender,
    create_judge,
    BaseAttacker,
    BaseDefender,
    TransferAttacker,
    BaseJudgeConfig,
    BaseJudge,
)
from jailbreakpipe.role import BaseAttackerConfig, BaseDefenderConfig


@dataclass
class InferPipelineConfig:
    """
    Configuration class for the inference pipeline.

    配置类，用于推理流程。

    :param attacker_config: Configuration for the attacker, detailing its parameters and behavior.
                            攻击者的配置，详细描述其参数和行为。
    :param defender_config: Configuration for the defender, detailing its parameters and behavior.
                            防御者的配置，详细描述其参数和行为。
    :param judge_configs: List of configurations for any judges used in evaluating the effectiveness of the defense.
                          评估防御有效性的评审者配置列表。
    """

    attacker_config: BaseAttackerConfig = field(default=None)
    defender_config: BaseDefenderConfig = field(default=None)
    judge_configs: List[BaseJudgeConfig] = field(default_factory=list)


def llm_register(attr: Union[BaseAttacker, BaseDefender, BaseJudge]) -> List[BaseLLM]:
    """
    Register any LLM instances associated with the given attribute (attacker, defender, or judge).

    为给定属性（攻击者、防御者或评审者）注册任何LLM实例。

    :param attr: An instance of an attacker, defender, or judge.
                 攻击者、防御者或评审者的实例。
    :return: List of registered LLM instances.
             注册的LLM实例列表。
    """
    results = []
    for name, value in vars(attr).items():
        if isinstance(value, BaseLLM):
            results.append(value)
    return results


class InferPipeline:
    """
    Inference pipeline for managing and executing attacks and defenses.

    管理和执行攻击和防御的推理流程。

    :param config: Configuration for the inference pipeline, including attacker, defender, and judges.
                   推理流程的配置，包括攻击者、防御者和评审者。
    :param verbose: Whether to enable verbose logging for debugging and transparency.
                    是否启用详细日志记录，用于调试和透明化。
    """

    def __init__(self, config: InferPipelineConfig, verbose: bool = False):
        # Create instances of the attacker, defender, and any judges based on the provided configurations.
        # 根据提供的配置创建攻击者、防御者和评审者的实例。
        self.attacker = create_attacker(config.attacker_config)
        self.defender = create_defender(config.defender_config)
        self.judges = [
            create_judge(judge_config) for judge_config in config.judge_configs
        ]
        # Register LLMs used by the attacker, defender, and judges.
        # 注册攻击者、防御者和评审者所使用的LLM。
        self.attack_llms = llm_register(self.attacker)
        self.defense_llms = llm_register(self.defender)
        self.judges_llms = [llm_register(judge) for judge in self.judges]
        self.verbose = verbose

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Execute the full inference process, including attack and defense phases.

        执行完整的推理过程，包括攻击和防御阶段。

        :param messages: Input messages to be processed through the pipeline.
                         要通过流程处理的输入消息。
        :param kwargs: Additional keyword arguments, such as reformulated requests for transfer attacks.
                       额外的关键字参数，例如用于转移攻击的重构请求。
        :return: A dictionary containing the final defended messages and the token usage details.
                 包含最终防御消息和Token使用详情的字典。
        """
        # Check if the attacker is a TransferAttacker and validate the required argument.
        # 检查攻击者是否为TransferAttacker，并验证所需参数。
        if isinstance(self.attacker, TransferAttacker):
            assert (
                "request_reformulated" in kwargs
            ), "TransferAttacker requires a 'request_reformulated' parameter."

        self.log(messages, "[INPUT]")

        # Perform the attack phase using the attacker instance.
        # 使用攻击者实例执行攻击阶段。
        attack = self.attacker.attack(
            messages, request_reformulated=kwargs.get("request_reformulated", None)
        )
        self.log(attack, "[ATTACK]")

        # Perform the defense phase using the defender instance.
        # 使用防御者实例执行防御阶段。
        defense = self.defender.defense(attack)
        self.log(defense, "[DEFENSE]")

        # Calculate the token usage for both phases.
        # 计算攻击和防御阶段的Token使用量。
        usage = self.calc_tokens()

        return {"messages": defense, "usage": usage}

    def parallel_judging(
        self, defense: List[Dict[str, str]], request: str
    ) -> Dict[str, Any]:
        """
        Perform parallel evaluation of the defense using multiple judges.

        使用多个评审者并行评估防御。

        :param defense: Messages that have been defended and need evaluation.
                        已经防御并需要评估的消息。
        :param request: The original request for context in evaluation.
                        评估中使用的原始请求。
        :return: A dictionary containing the results from each judge.
                 包含每个评审者结果的字典。
        """
        judge_results = {}

        # Use ThreadPoolExecutor for parallel judge evaluations.
        # 使用ThreadPoolExecutor进行评审者的并行评估。
        with ThreadPoolExecutor() as executor:
            future_to_judge = {
                executor.submit(judge.judge, defense, request): judge
                for judge in self.judges
            }
            for future in as_completed(future_to_judge):
                judge = future_to_judge[future]
                try:
                    result = future.result()
                    judge_results[judge._NAME] = result
                except Exception as exc:
                    print(f"Judge {judge._NAME} generated an exception: {exc}")

        return judge_results

    def calc_tokens(self) -> Dict[str, Dict[str, int]]:
        """
        Calculate the token usage for both the attacker and defender, summarizing prompt and completion tokens.

        计算攻击者和防御者的Token使用量，包括提示和完成Token。

        :return: Dictionary containing detailed token usage for both roles.
                 包含双方详细Token使用量的字典。
        """
        # Calculate token usage for the attacker.
        # 计算攻击者的Token使用量。
        attacker_prompt_tokens = sum(llm.prompt_tokens for llm in self.attack_llms)
        attacker_completion_tokens = sum(
            llm.completion_tokens for llm in self.attack_llms
        )

        # Calculate token usage for the defender.
        # 计算防御者的Token使用量。
        defender_prompt_tokens = sum(llm.prompt_tokens for llm in self.defense_llms)
        defender_completion_tokens = sum(
            llm.completion_tokens for llm in self.defense_llms
        )

        return {
            "attacker": {
                "prompt_tokens": attacker_prompt_tokens,
                "completion_tokens": attacker_completion_tokens,
            },
            "defender": {
                "prompt_tokens": defender_prompt_tokens,
                "completion_tokens": defender_completion_tokens,
            },
        }

    def reset(self):
        """
        Reset the state of all LLMs in the pipeline, preparing them for a new inference run.

        重置流程中所有LLM的状态，准备进行新的推理。
        """
        for llm in self.attack_llms + self.defense_llms:
            llm.reset()

    def log(self, messages: List[Dict[str, str]], stage: str = None):
        """
        Log messages for each stage of the pipeline if verbose logging is enabled.

        如果启用了详细日志记录，则记录流程每个阶段的消息。

        :param messages: The messages to be logged at the current stage.
                         当前阶段要记录的消息。
        :param stage: A label for the current stage of the pipeline.
                      流程当前阶段的标签。
        """
        if self.verbose:
            pass
            # print(stage)
            # print(messages)
            # print()
