# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/10/11 21:47
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : alpaca_inference.py
# explain   :  Run inference for alpaca evaluation using a specific LLM and defense configuration.

import os
import json
import pandas as pd
import yaml
import argparse
import warnings
import logging

from tqdm import tqdm

from panda_guard.llms import HuggingFaceLLMConfig, create_llm
from panda_guard.pipelines.inference import InferPipeline, InferPipelineConfig
from panda_guard.utils import parse_configs_from_dict, wait_for_gpu_memory

from datasets import load_dataset


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Alpaca evaluation inference")
    parser.add_argument(
        '--config',
        type=str,
        default="../../configs/tasks/alpaca_eval.yaml",
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--llm',
        type=str,
        required=False,
        help='Path to the target LLM file'
    )
    parser.add_argument(
        '--llm-gen',
        type=str,
        default='../../configs/defenses/llm_gen/alpaca_eval.yaml',
        help='Path to the LLM generator file'
    )
    parser.add_argument(
        '--defense',
        type=str,
        required=True,
        help='Path to the defense configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="../../benchmarks/alpaca_eval",
        help='Output directory'
    )
    parser.add_argument(
        '--repeats',
        type=int, default=1,
        help='Number of times to repeat the experiment'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run the model on'
    )
    parser.add_argument(
        '--max-queries',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--visible',
        action='store_true'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default="WARNING",
        help="Logging level, e.g., DEBUG, INFO, WARNING, ERROR"
    )
    return parser.parse_args()


def run_inference(args):
    logging.basicConfig(level=eval(f'logging.{args.log_level}'), format='%(asctime)s - %(levelname)s - %(message)s')

    config_dict = load_yaml(args.config)
    config_dict["defender"] = load_yaml(args.defense)

    if args.visible:
        logging.info(f"Loaded configuration: {config_dict}")

    llm_dict = load_yaml(args.llm)
    llm_gen_dict = load_yaml(args.llm_gen)

    if llm_dict:
        config_dict["defender"]["target_llm_config"] = llm_dict
    else:
        llm_dict = config_dict["defender"]["target_llm_config"]

    if args.device:
        config_dict["defender"]["target_llm_config"]["device_map"] = args.device

    if llm_gen_dict:
        config_dict["defender"]["target_llm_gen_config"].update(llm_gen_dict)

    output_file = os.path.join(
        args.output_dir,
        f"{llm_dict['model_name'].replace('/', '_').replace('/', '_')}",
        f'{config_dict["defender"]["defender_cls"]}',
        "results.json"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        logging.warning(f"Output file {output_file} already exists. Skipping.")
        return

    yaml.dump(config_dict, open(output_file.replace('results.json', 'config.yaml'), 'w'))

    # Initialize the pipeline
    attacker_config, defender_config, _ = parse_configs_from_dict(config_dict)
    pipe = InferPipeline(
        InferPipelineConfig(
            attacker_config=attacker_config,
            defender_config=defender_config,
        ),
        verbose=False
    )

    # Load evaluation dataset
    eval_set = load_dataset("/root/.cache/huggingface/datasets/tatsu-lab___alpaca_eval")['test']
    # eval_set = load_dataset("/home/floyed/.cache/huggingface/datasets/tatsu-lab___alpaca_eval")['test']
    
    # 去除instruction重复的样本，只保留第一次出现的
    seen_instructions = set()
    unique_indices = []
    duplicate_count = 0

    for i, item in enumerate(eval_set):
        instruction = item.get('instruction', '')
        if instruction not in seen_instructions:
            seen_instructions.add(instruction)
            unique_indices.append(i)
        else:
            duplicate_count += 1

    # 选择唯一的样本
    eval_set = eval_set.select(unique_indices)
    
    # Run inference for each row
    if args.max_queries:
        eval_set = eval_set.select(range(min(len(eval_set), args.max_queries)))

    iterator = tqdm(eval_set, total=len(eval_set), desc=args.llm.split('/')[-1]) if args.visible else eval_set
    results = []
    for row in iterator:
        result = pipe(
            [{"role": "user", "content": row["instruction"]}],
            request_reformulated=row["instruction"]
        )
        row["output"] = result["messages"][-1]["content"]
        row["generator"] = config_dict["defender"]["target_llm_config"]["model_name"].replace('/', '_')
        results.append(row)
        pipe.reset()

    # Save results
    with open(output_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    logging.info(f"Results saved to {output_file}")


if __name__ == '__main__':
    args = parse_args()
    if not args.visible:
        warnings.filterwarnings("ignore")
    run_inference(args)
