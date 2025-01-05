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
from jailbreakpipe.role.attacks.pair import PairAttacker, PairAttackerConfig
from jailbreakpipe.role.attacks.attacker_registry import create_attacker, ATTACKERS
from jailbreakpipe.role.attacks.gcg import GCGAttacker, GCGAttackerConfig
from jailbreakpipe.role.attacks.tap import TAPAttacker, TAPAttackerConfig
from jailbreakpipe.role.attacks.autodan.autodan import AutoDanAttacker, AutoDanAttackerConfig

print(PairAttacker.__class__)

__all__ = [
    "BaseAttacker",
    "BaseAttackerConfig",
    "TransferAttacker",
    "TransferAttackerConfig",
    "RewriteAttacker",
    "RewriteAttackerConfig",
    "PairAttacker",
    "PairAttackerConfig",
    "GCGAttacker",
    "GCGAttackerConfig",
    "create_attacker",
    "TAPAttacker",
    "TAPAttackerConfig",
    "AutoDanAttacker",
    "AutoDanAttackerConfig",
]
