# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 20:56
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : __init__.py.py
# explain   :

from panda_guard.llms.base import BaseLLM, BaseLLMConfig, LLMGenerateConfig

from panda_guard.utils import ComponentRegistry

llm_registry = ComponentRegistry[BaseLLM]("llm", BaseLLM, "panda_guard.llms")

config_registry = ComponentRegistry[BaseLLMConfig](
    "llm_config", BaseLLMConfig, "panda_guard.llm_configs"
)


def create_llm(config: BaseLLMConfig) -> BaseLLM:

    config_class_name = config.__class__.__name__
    if config_class_name.endswith("Config"):
        llm_class_name = config_class_name[:-6]
        try:
            llm_class = llm_registry.get_component_class(llm_class_name)
            return llm_class(config)
        except ValueError:
            pass

    raise ValueError(f"Cannot Create LLM from Config: {config}")


LLMS = {}


def __getattr__(name):
    try:
        if name.endswith("Config"):
            return config_registry.get_component_class(name)
        else:
            llm_class = llm_registry.get_component_class(name)
            if name not in LLMS:
                LLMS[name] = llm_class
            return llm_class
    except ValueError:
        raise AttributeError(f"module 'panda_guard.llms' has no attribute '{name}'")
