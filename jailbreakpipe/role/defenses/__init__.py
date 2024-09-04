# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 20:57
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : __init__.py.py
# explain   :

from jailbreakpipe.role.defenses.base import BaseDefender, BaseDefenderConfig, REJECT_RESPONSE
from jailbreakpipe.role.defenses.rewrite import SelfReminderDefender, SelfReminderDefenderConfig, NoneDefender, NoneDefenderConfig
from jailbreakpipe.role.defenses.icl import IclDefender, IclDefenderConfig
from jailbreakpipe.role.defenses.smoothllm import SmoothLLMDefender, SmoothLLMDefenderConfig
from jailbreakpipe.role.defenses.semantic_smoothllm import SemanticSmoothLLMDefender, SemanticSmoothLLMDefenderConfig
from jailbreakpipe.role.defenses.paraphrase import ParaphraseDefender, ParaphraseDefenderConfig
from jailbreakpipe.role.defenses.back_translate import BackTranslationDefender, BackTranslationDefenderConfig
from jailbreakpipe.role.defenses.perplexity_filter import PerplexityFilterDefender, PerplexityFilterDefenderConfig
from jailbreakpipe.role.defenses.defender_registry import create_defender

__all__ = [
    "BaseDefender",
    "BaseDefenderConfig",
    "REJECT_RESPONSE",
    "NoneDefender",
    "NoneDefenderConfig",
    "SelfReminderDefender",
    "SelfReminderDefenderConfig",
    "IclDefender",
    "IclDefenderConfig",
    "SmoothLLMDefender",
    "SmoothLLMDefenderConfig",
    "SemanticSmoothLLMDefender",
    "SemanticSmoothLLMDefenderConfig",
    "ParaphraseDefender",
    "ParaphraseDefenderConfig",
    "BackTranslationDefender",
    "BackTranslationDefenderConfig",
    "PerplexityFilterDefender",
    "PerplexityFilterDefenderConfig",
    "create_defender",
]
