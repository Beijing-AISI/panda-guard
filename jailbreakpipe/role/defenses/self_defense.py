from typing import Dict, List
from dataclasses import dataclass, field
from jailbreakpipe.role.defenses import BaseDefender, BaseDefenderConfig
from jailbreakpipe.role.defenses.defender_registry import register_defender
from jailbreakpipe.llms import BaseLLMConfig, LLMGenerateConfig
from jailbreakpipe.role.judges import PairLLMJudgeConfig, BaseJudgeConfig, create_judge
import torch.nn.functional as F
from jailbreakpipe.utils import load_class, parse_nested_config


@dataclass
class SelfDefenseDefenderConfig(BaseDefenderConfig):
    """
    Configuration for the Base Defender.

    :param defender_cls: Class of the defender.
    :type defender_cls: str
    :param defender_name: Name of the defender.
    :type defender_name: str
    :param target_llm_config: Configuration for the target language model.
    :type target_llm_config: BaseLLMConfig
    :param target_llm_gen_config: Configuration for generating output with the target LLM.
    :type target_llm_gen_config: LLMGenerateConfig
    :param judge_config: Configuration for the judge, used to assess the defense.
    :type judge_config: BaseJudgeConfig
    :param judge_score_threshold: The threshold score for the judge to decide the defense action.
    :type judge_score_threshold: int
    """

    defender_cls: str = field(default="SelfDefenseDefender")
    defender_name: str = field(default="SelfDefense")
    target_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    target_llm_gen_config: LLMGenerateConfig = field(default=None)

    judge_config: BaseJudgeConfig = field(default_factory=PairLLMJudgeConfig)
    judge_score_threshold: int = field(default=5)


@register_defender
class SelfDefenseDefender(BaseDefender):
    """
    SelfDefenseDefender implements the defense mechanism for the attack,
    generating responses and judging their appropriateness.
    https://arxiv.org/abs/2308.07308
    """

    def __init__(self, config: SelfDefenseDefenderConfig):
        """
        Initialize the defender with the provided configuration.

        :param config: Configuration for the SelfDefenseDefender.
        :type config: SelfDefenseDefenderConfig
        """
        # create self.target_llm, self.target_llm_gen_config
        super().__init__(config)

        judge_config_dict = config.judge_config
        judge_config_class = load_class(judge_config_dict.get("judge_cls"))
        judge_config = parse_nested_config(judge_config_class, judge_config_dict)
        self.judge = create_judge(config=judge_config)
        self.judge_score_threshold = config.judge_score_threshold

    def response(self, message: List[Dict[str, str]]):
        """
        Generate a response from the target language model.

        :param message: The input message to be processed by the target language model.
        :type message: List[Dict[str, str]]
        :return: Generated response based on the input message.
        :rtype: List[Dict[str, str]]
        """
        message = self.target_llm.generate(
            messages=message, config=self.target_llm_gen_config
        )
        return message

    def defense(self, messages=List[Dict[str, str]]):
        """
        Perform defense by generating a response, judging it, and applying defense action.

        :param messages: The input messages that are subject to defense.
        :type messages: List[Dict[str, str]]
        :return: The final defense response after evaluating the defense.
        :rtype: List[Dict[str, str]]
        """

        # without response
        naive_response = self.response(message=messages[:])

        # judge
        judge_score = self.judge.judge(messages=naive_response)

        if judge_score >= self.judge_score_threshold:
            naive_response[-1]["content"] = "Sorry, I CANNOT assist with that."

        return naive_response
