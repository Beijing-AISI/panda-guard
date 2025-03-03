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
from .cold_attack.util import *
from .cold_attack.bleuloss import batch_log_bleulosscnn_ae
from .cold_attack.attack_suffix import attack_generation

stop_words = set(stopwords.words('english'))

def get_fixed_args():
    # 原文中配置为: batch_size=8, length=20, max_length=20, num_iters=2000, goal_weight=100
    args = argparse.Namespace(
        no_cuda=False,
        verbose=False,
        print_every=2000,
        pretrained_model="llama2",
        wandb=False,
        straight_through=True,
        topk=10,
        rl_topk=0,
        lexical='max',
        lexical_variants=False,
        if_zx=False,
        fp16=True,
        version="",
        start=0,
        end=50,
        mode='suffix',
        control_type='sentiment',
        batch_size=1,
        length=100,
        max_length=100,
        frozen_length=0,
        goal_weight=500,
        rej_weight=100,
        abductive_filterx=False,
        lr_nll_portion=1.0,
        prefix_length=0,
        counterfactual_max_ngram=3,
        no_loss_rerank=False,
        use_sysprompt=False,
        input_lgt_temp=1.0,
        output_lgt_temp=1.0,
        # temperature
        rl_output_lgt_temp=1.0,
        init_temp=1,
        init_mode='original',
        # lr
        stepsize=0.1,
        stepsize_ratio=1,
        stepsize_iters=1000,
        # iterations
        num_iters=1000,
        min_iters=0,
        noise_iters=1,
        win_anneal_iters=1000,
        constraint_iters=1000,
        # gaussian noise
        gs_mean=0.0,
        gs_std=0.01,
        large_noise_iters="50, 200, 500, 1500",
        large_gs_std="0.1, 0.05, 0.01, 0.001",
    )
    return args



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

    实现过程中不同的一些点：
    1. target 在框架中没有传入，简单使用'Sure, here is'
    2. 原文中需要使用GPT-4从多个sample中取出合适的一个. 为了不使用GPT-4. 我们将batch size设置为1, 只用第一个.

    :param config: Configuration for the ColdAttacker.  用于ColdAttacker的配置
    """

    def __init__(self, config: ColdAttackerConfig):
        super().__init__(config)

        self.config = config

        self.llm = create_llm(config=config.white_box_llm_config)
        # TOOD: 这里的所有的config, 应该都写到BaseAttackerConfig中, 并能使用 .yaml 文件进行配置 / 读取. 
        self.args = get_fixed_args()

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
