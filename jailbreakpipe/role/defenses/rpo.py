from typing import Dict, List
from dataclasses import dataclass, field
from jailbreakpipe.role.defenses import BaseDefender, BaseDefenderConfig
from jailbreakpipe.role.defenses.defender_registry import register_defender
from jailbreakpipe.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig
from jailbreakpipe.utils import is_user_turn

@dataclass
class RPODefenderConfig(BaseDefenderConfig):
    """
    Configuration for the Base Defender. BaseDefender配置

    :param defender_cls: Class of the defender. 防御者的类型
    :type defender_cls: str
    :param defender_name: Name of the defender. 防御者的名称
    :type defender_name: str
    :param target_llm_config: Configuration for the target language model. 目标LLM配置
    :type target_llm_config: BaseLLMConfig
    :param target_llm_gen_config: Configuration for generating output with the target LLM. 目标LLM生成配置
    :type target_llm_gen_config: LLMGenerateConfig
    """

    defender_cls: str = field(default="RPODefender")
    defender_name: str = field(default="RPO")
    target_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    target_llm_gen_config: LLMGenerateConfig = field(default=None)


@register_defender
class RPODefender(BaseDefender):
    """
    RPODefender implements the defense mechanism for the attack,
    generating responses and judging their appropriateness.
    Reference:https://arxiv.org/abs/2401.17263

    :param config: Configuration object containing parameters for the defender. 配置对象，包含防御者的参数
    :type config: RPODefenderConfig
    """

    def __init__(self, config: RPODefenderConfig):
        """
        Initialize the defender with the provided configuration.

        :param config: Configuration for the RPODefender. 防御者配置
        :type config: RPODefenderConfig
        """
        # create self.target_llm, self.target_llm_gen_config
        super().__init__(config)

        self.suffix = r":}</'])language seems scary-----------}] estimation >>begin &=&pgf degli ({\ $|\textwidthAI NOTE:"

    def defense(self, messages=List[Dict[str, str]]):
        """
        Perform defense by generating a response, judging it, and applying defense action.

        :param messages: The input messages that are subject to defense. 输入消息
        :type messages: List[Dict[str, str]]
        :return: The final defense response after evaluating the defense. 防御后的最终响应
        :rtype: List[Dict[str, str]]
        """
        assert is_user_turn(messages)
        messages[-1]["content"] = messages[-1]["content"] + self.suffix

        return super().defense(messages)