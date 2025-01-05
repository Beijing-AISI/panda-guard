# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 20:57
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : __init__.py.py
# explain   :

from jailbreakpipe.role.attacks.base import BaseAttacker, BaseAttackerConfig
from jailbreakpipe.role.attacks.transfer import TransferAttacker, TransferAttackerConfig
from jailbreakpipe.role.attacks.rewrite import RewriteAttacker, RewriteAttackerConfig
from jailbreakpipe.role.attacks.art_prompt import ArtPromptAttacker, ArtPromptAttackerConfig
from jailbreakpipe.role.attacks.advprompter import AdvprompterAttacker,AdvprompterAttackerConfig
from jailbreakpipe.role.attacks.attacker_registry import create_attacker, ATTACKERS

__all__ = [
    "BaseAttacker",
    "BaseAttackerConfig",
    "TransferAttacker",
    "TransferAttackerConfig",
    "RewriteAttacker",
    "RewriteAttackerConfig",
    "ArtPromptAttacker",
    "ArtPromptAttackerConfig",
    "AdvprompterAttacker",
    "AdvprompterAttackerConfig",
    "create_attacker",
]
