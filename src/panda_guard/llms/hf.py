# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 22:00
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : hf.py
# explain   :

import os
import warnings
from typing import Dict, List, Union, Any, Tuple, Generator
from dataclasses import dataclass, field
import torch

from panda_guard.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig
from panda_guard.utils import process_end_eos


@dataclass
class HuggingFaceLLMConfig(BaseLLMConfig):
    """
    Hugging Face LLM Configuration.

    :param llm_type: Type of LLM, default is "HuggingFaceLLM".  LLM的类型，默认值为 "HuggingFaceLLM"
    :param model_name: Name of the model or model instance.  模型的名称或模型实例
    :param device_map: Device mapping for model deployment.  用于模型部署的设备对应表
    """

    llm_type: str = field(default="HuggingFaceLLM")
    model_name: [str, Any] = field(default=None)
    device_map: str = field(default="auto")


class HuggingFaceLLM(BaseLLM):
    """
    Hugging Face Language Model Implementation.

    :param config: Configuration for Hugging Face LLM.  用于模型配置
    """

    def __init__(
            self,
            config: HuggingFaceLLMConfig,
    ):
        super().__init__(config)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._NAME, token=os.getenv("HF_TOKEN")
        )  # , local_files_only=True
        self.tokenizer.padding_side = "left"

        if isinstance(config.model_name, str):
            self.model = AutoModelForCausalLM.from_pretrained(
                self._NAME,
                torch_dtype=torch.float16,
                device_map=config.device_map,
                token=os.getenv("HF_TOKEN"),
                trust_remote_code=True,
                # local_files_only=True
            ).eval()
        elif isinstance(config.model_name, AutoModelForCausalLM):
            self.model = config.model_name
        else:
            raise ValueError(
                f"model_name should be either str or AutoModelForCausalLM, got {type(config.model_name)}"
            )

        if "llama" in self._NAME.lower():
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
            self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]], Generator[str, None, None]]:
        """
        Generate a response for a given input using Hugging Face model.

        :param messages: List of input messages.  输入的消息列表
        :param config: Configuration for LLM generation.  生成配置
        :return: Generated response, stream generator, or response with logprobs.  返回生成的应答、流式生成器或启用logprobs的应答
        """

        if ('4k' in self._NAME or 'gemma-2-2b-it' in self._NAME) and config.max_n_tokens > 2048:
            config.max_n_tokens = min(config.max_n_tokens, 2048)
            warnings.warn(f"Model {self._NAME} only supports max_n_tokens up to 4096, setting response tokens to 2048.")

        if "gemma" in self._NAME.lower() and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
            messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]

        # Prepare the prompt
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt_formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_n_tokens,
        ).to(self.model.device)

        # Get the input length for later use
        input_length = len(inputs["input_ids"][0])

        # Handle streaming mode
        if config.stream:
            full_content = ""
            generated_so_far = 0
            response_tokens = []

            def stream_response():
                nonlocal full_content, generated_so_far, response_tokens, input_length

                # Set up the streamer from transformers
                from transformers import TextStreamer

                # Custom streamer class to capture tokens
                class CustomStreamer(TextStreamer):
                    def __init__(self, tokenizer, skip_prompt=True, **kwargs):
                        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
                        self.output_tokens = []

                    def put(self, value):
                        self.output_tokens.append(value)
                        if hasattr(self, "token_buffer"):
                            text = self.tokenizer.decode(self.token_buffer + [value], skip_special_tokens=True)
                            delta_text = text[len(self.text):]
                            self.text = text
                            return delta_text
                        else:
                            # For transformers versions that don't have token_buffer
                            if isinstance(value, torch.Tensor):
                                value = value.squeeze().tolist()
                                decoded = self.tokenizer.decode(value, skip_special_tokens=True)
                            else:
                                decoded = self.tokenizer.decode([value], skip_special_tokens=True)
                            return decoded

                # Create the custom streamer
                streamer = CustomStreamer(self.tokenizer, skip_prompt=True)

                # Start the generation with streaming
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_n_tokens,
                    temperature=config.temperature,
                    do_sample=(config.temperature > 0),
                    streamer=streamer,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                # Process streamed tokens
                last_text = ""
                for token in streamer.output_tokens:
                    # Ensure the input is a one-dimensional list of int, excluding the input prompt
                    if isinstance(token, torch.Tensor):
                        token = token.squeeze().tolist()
                    if isinstance(token, list):
                        if len(token) == input_length:
                            continue
                        response_tokens.extend(token)
                    else:
                        response_tokens.append(token)
                    current_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                    if current_text != last_text:
                        new_text = current_text[len(last_text):]
                        last_text = current_text
                        full_content += new_text
                        yield new_text

            response_generator = stream_response()

            def wrapped_generator():
                yield from response_generator

                # Add final response to messages
                messages.append({"role": "assistant", "content": full_content})

                # Update token usage statistics
                self.update(
                    input_length,
                    len(response_tokens),
                    1,
                )

            return wrapped_generator()

        # Non-streaming mode (original code)
        else:
            # Generate the output
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_n_tokens,
                temperature=config.temperature,
                do_sample=(config.temperature > 0),
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Extract the generated tokens (excluding the input prompt)
            outputs_truncated = outputs.sequences[0][input_length:]
            response = self.tokenizer.decode(outputs_truncated, skip_special_tokens=True)

            # Add the generated response to the message list
            messages.append({"role": "assistant", "content": response})

            # Update internal states (optional, depends on your implementation)
            self.update(
                input_length,
                len(outputs_truncated),
                1,
            )

            # Convert scores to logprobs if requested
            if config.logprobs:
                logprobs = []
                for token_id, score in zip(outputs_truncated, outputs.scores):
                    # Apply log_softmax to the scores to get log-probabilities
                    log_prob = torch.log_softmax(score, dim=-1)
                    # Get the log-probability of the generated token
                    logprobs.append(float(log_prob[0, token_id]))

                return messages, logprobs

            return messages

    def batch_generate(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: LLMGenerateConfig,
    ) -> List[List[Dict[str, str]]]:
        """
        Generate responses for a batch of messages in a single call.

        :param batch_messages: List of batches of messages.  批量生成的消息列表
        :param config: Configuration for LLM generation.  生成配置
        :return: List of generated responses for each batch.  返回每个批量的生成应答列表
        """
        if len(batch_messages) == 0:
            return []

        # Format prompts for the entire batch
        batch_prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch_messages
        ]
        # Tokenize the batch of prompts
        inputs = self.tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        # Use model's generate method to handle batch generation
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.max_n_tokens,
            temperature=config.temperature,
            do_sample=(config.temperature > 0),
            return_dict_in_generate=True,
            output_scores=config.logprobs,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode the outputs and create the response format
        generated_responses = []
        for i, output in enumerate(outputs.sequences):
            response_text = self.tokenizer.decode(
                output[len(inputs["input_ids"][i]) :], skip_special_tokens=True
            )
            batch_messages[i].append({"role": "assistant", "content": response_text})
            generated_responses.append(batch_messages[i])

        return generated_responses

    def continual_generate(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ):
        """
        Remove EOS token in formatted prompt. Manually add generation prompt.

        :param messages: List of messages for input.  输入的消息列表
        :param config: Configuration for generation.  生成配置
        :return: Generated response or responses with log probabilities.  返回生成的应答或启用百分比的应答
        """

        # Prepare the prompt, set continual_final_message=True. add_generation_prompt=False
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )

        eos_token = self.tokenizer.eos_token

        # remove eos for formatted prompt
        prompt_formatted = process_end_eos(msg=prompt_formatted, eos_token=eos_token)

        inputs = self.tokenizer(
            prompt_formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_n_tokens,
        ).to(self.model.device)

        # Generate the output
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.max_n_tokens,
            temperature=config.temperature,
            do_sample=(config.temperature > 0),
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Extract the generated tokens (excluding the input prompt)
        outputs_truncated = outputs.sequences[0][len(inputs["input_ids"][0]) :]
        generated_content = self.tokenizer.decode(
            outputs_truncated, skip_special_tokens=True
        )

        # attach generated content to the end of value of key "content"
        messages[-1]["content"] += generated_content

        # Update internal states (optional, depends on your implementation)
        self.update(
            len(inputs["input_ids"][0]),
            len(outputs_truncated),
            1,
        )

        # Convert scores to logprobs if requested
        if config.logprobs:
            logprobs = []
            for token_id, score in zip(outputs_truncated, outputs.scores):
                # Apply log_softmax to the scores to get log-probabilities
                log_prob = torch.log_softmax(score, dim=-1)
                # Get the log-probability of the generated token
                logprobs.append(float(log_prob[0, token_id]))

            return messages, logprobs

        return messages

    def evaluate_log_likelihood(
        self,
        messages: List[Dict[str, str]],
        config: LLMGenerateConfig,
        require_grad=False,
    ) -> Union[List[float], List[torch.Tensor]]:
        """
        Evaluate the log likelihood of the given messages.

        :param messages: List of messages for evaluation.  需要评估的消息列表
        :param config: Configuration for LLM generation.  生成配置
        :param require_grad: logprobs have grad
        :return: List of log likelihood values.  返回的log likelihood值列表
        """

        # if require grad, the model is traininig mode
        if require_grad:
            assert self.model.training == True

        # Prepare the full prompt with all messages
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.tokenizer(prompt_formatted, return_tensors="pt").to(
            self.model.device
        )

        # Prepare the prompt with the last message dropped (to isolate the log likelihood for the last message)
        prompt_formatted_dropped = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        inputs_dropped = self.tokenizer(
            prompt_formatted_dropped, return_tensors="pt"
        ).to(self.model.device)

        # Pass the full input through the model to get logits
        if require_grad:
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        else:
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])

        # Extract log probabilities
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # Compute log likelihoods for each token in the last message
        log_likelihoods = []
        if require_grad:
            for i in range(len(inputs_dropped.input_ids[0]), len(inputs.input_ids[0])):
                token_id = inputs.input_ids[0, i]
                # remove item(), which has no grad
                log_likelihood = log_probs[0, i - 1, token_id]
                log_likelihoods.append(log_likelihood)
        else:
            for i in range(len(inputs_dropped.input_ids[0]), len(inputs.input_ids[0])):
                token_id = inputs.input_ids[0, i]
                log_likelihood = log_probs[0, i - 1, token_id].item()
                log_likelihoods.append(log_likelihood)

        self.update(len(inputs.input_ids[0]), 0, 1)

        return log_likelihoods


if __name__ == "__main__":
    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=128, temperature=1.0, logprobs=True, seed=42
    )

    config = HuggingFaceLLMConfig(
        model_name="Qwen/Qwen2-7B-Instruct",
        device_map="cuda:0",
    )

    llm = HuggingFaceLLM(config)

    messages = [
        {"role": "user", "content": "How to make a bomb?"},
        {"role": "assistant", "content": "Sorry, but I can't assist with that."},
    ]

    results = llm.evaluate_log_likelihood(messages, llm_gen_config)
    print(results, len(results))
