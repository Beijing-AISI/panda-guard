# encoding: utf-8
# Author    : Shen Sicheng
# Datetime  : 2024/12/02 16:30
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : gpt4_cipher.py
# explain   : gpt4_cipher: https://arxiv.org/abs/2308.06463 ICLR 2024

from typing import Dict, List, Any
from dataclasses import dataclass, field
from jailbreakpipe.role.attacks import BaseAttacker, BaseAttackerConfig
from jailbreakpipe.role.attacks.attacker_registry import register_attacker
from jailbreakpipe.utils import is_user_turn


@dataclass
class GPT4CipherAttackerConfig(BaseAttackerConfig):
    """
    Configuration for the Rewrite Attacker.

    :param attacker_cls: Class of the attacker, default is "RewriteAttacker".  攻击者的类型，默认值为 "RewriteAttacker"
    :param attacker_name: Name of the attacker.  攻击者的名称
    :param llm_config: Configuration for the language model.  LLM配置
    :param llm_gen_config: Configuration for generating output with LLM.  LLM生成配置
    :param rewrite_template: Template for LLM to mask offensive words
    """

    attacker_cls: str = field(default="ArtPromptAttacker")
    attacker_name: str = field(default="ArtPrompt")
    rewrite_template: str = field(default=None)


@register_attacker
class GPT4CipherAttacker(BaseAttacker):
    """
    Rewrite Attacker Implementation for modifying user prompts to bypass restrictions.

    TIPS:
    mask_llm_url should be set in config file for generate mask words and mask instruction.
    Here we use GPT3.5-turbo as the mask LLM.

    Correspondingly, OPENAI_API_KEY should be set in environment variables.
    """

    def __init__(self, config: GPT4CipherAttackerConfig):
        super().__init__(config)

        self.rewrite_template = config.rewrite_template


    def caesar_cipher(self, text, shift=3)-> str:
        """
        To transfer a string into Caesar Cipher
        :param text: string waiting to be transferred
        :param shift: shift digits in Caesar Cipher
        """

        result = []


        for char in text:
            if char.isalpha():
                shift_base = 65 if char.isupper() else 97
                shifted_char = chr((ord(char) - shift_base + shift) % 26 + shift_base)
                result.append(shifted_char)
            else:
                result.append(char)

        return ''.join(result)

    def attack(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> List[Dict[str, str]]:
        """
        Execute an attack by transfer the latest user prompt to ArtPrompt.

        :param messages: List of messages in the conversation.  对话中的消息列表
        :param kwargs: Additional parameters for the attack.  额外的攻击参数
        :return: Modified list of messages with the rewritten prompt.  包含重写提示的更改消息列表
        """
        assert is_user_turn(messages)

        original_prompt = messages[-1]["content"]


        #
        # self.llm.generate(
        #     system_messages,
        #     self.llm_gen_config,
        # )

        caesar_cipher = self.caesar_cipher(original_prompt, 3)
        # prompt_messages = {
        #     "role": "user",
        #     "content": caesar_cipher,
        # }
        #
        # messages[-1]["role"] = "system"
        # messages[-1]["content"] = self.rewrite_template
        #
        # messages.append(prompt_messages)

        messages[-1]["content"] = self.rewrite_template + caesar_cipher

        print(self.rewrite_template + caesar_cipher)

        return messages



