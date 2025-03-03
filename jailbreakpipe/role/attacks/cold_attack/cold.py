#!/usr/bin/env python
# coding: utf-8

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

import os
import numpy as np
import time
import argparse
import random
import sys

from nltk.corpus import stopwords
from transformers import AutoTokenizer
from jailbreakpipe.role.attacks.cold_attack.util import *
from jailbreakpipe.role.attacks.cold_attack.bleuloss import batch_log_bleulosscnn_ae
from jailbreakpipe.role.attacks.cold_attack.attack_suffix import attack_generation

stop_words = set(stopwords.words('english'))

@dataclass
class ColdAttackerConfig(BaseAttackerConfig):
    """
    Configuration for the Cold Attacker.

    :param attacker_cls: Class of the attacker, default is "ColdAttacker".  攻击者的类型，默认值为 "ColdAttacker"
    :param attacker_name: Name of the attacker.  攻击者的名称
    :param white_box_llm_config: Configuration of attacker llm. 白盒攻击时使用的llm配置
    :param white_box_llm_gen_config: Generation configuration for the attacker's LLM. 白盒攻击者 LLM 的生成配置
    """

    attacker_cls: str = field(default="ColdAttacker")
    attacker_name: str = field(default=None)
    white_box_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    white_box_llm_gen_config: LLMGenerateConfig = field(default=None)


@register_attacker
class ColdAttacker(BaseAttacker):
    """
    Cold Attacker Implementation that substitutes the user message with a pre-formulated attack prompt.
    Reference：Guo, Xingang and Yu, Fangxu and Zhang, Huan and Qin, Lianhui and Hu, Bin, 2024, Cold-attack: Jailbreaking llms with stealthiness and controllability, ICML 2024
    实现过程中不同的一些点：
    1. target 在框架中没有传入，简单使用'Sure, here is'
    2. 原文中需要使用GPT-4从多个sample中取出合适的一个. 为了不使用GPT-4. 我们将batch size设置为1, 只用第一个.

    :param config: Configuration for the ColdAttacker.  用于ColdAttacker的配置
    """

    def __init__(self, config: ColdAttackerConfig):
        super().__init__(config)

        self.config = config

        self.llm = create_llm(config=config.white_box_llm_config)
        self.args = config.cold_config

    def attack(self, messages: List[Dict[str, str]], **kwargs) -> List[Dict[str, str]]:
        """
        Execute an attack by transferring a reformulated request into the conversation.

        :param messages: List of messages in the conversation.  对话中的消息列表
        :param kwargs: Additional parameters for the attack, must include "request_reformulated".  额外攻击参数
        :return: Prompts containing harmful attacks on the target, is of the form “role: user, content: xx”. 含有目标的有害攻击的prompt, 是"role: user, content: xx的形式"
        """

        question = messages[0]["content"]

        device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"

        # Load pretrained model
        model, tokenizer = self.llm.model, self.llm.tokenizer

        # Freeze weights
        for param in model.parameters():
            param.requires_grad = False

        # cover configuration
        self.args.pretrained_model = self.config.white_box_llm_config.model_name

        if "suffix" in self.args.mode:
            messages = attack_generation(model, tokenizer, question, device, self.args)

        return messages
