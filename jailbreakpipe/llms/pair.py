import os
from typing import Dict, List, Union, Any, Tuple
from dataclasses import dataclass, field
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from jailbreakpipe.llms.llm_registry import register_llm
from jailbreakpipe.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig
from jailbreakpipe.llms import HuggingFaceLLMConfig


@register_llm
class HuggingFaceContinualLLM(BaseLLM):
    """
    Hugging Face Language Model Implementation.
    Set tokenizer.apply_chat_template 'continue_final_message=True'. Remove closure symbol of input message

    :param config: Configuration for Hugging Face LLM.  用于模型配置
    """

    def __init__(
        self,
        config: HuggingFaceLLMConfig,
    ):
        super().__init__(config)

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
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]:
        """
        Generate a response for a given input using Hugging Face model.

        :param messages: List of input messages.  输入的消息列表
        :param config: Configuration for LLM generation.  生成配置
        :return: Generated response or response with logprobs.  返回生成的应答或启用logprobs的应答
        """
        # Prepare the prompt, set continual_final_message=True. add_generation_prompt=False
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )

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
        response = self.tokenizer.decode(outputs_truncated, skip_special_tokens=True)

        # Add the generated response to the message list
        messages.append({"role": "assistant", "content": response})

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
                messages, tokenize=False, continue_final_message=True
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

    def evaluate_log_likelihood(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> List[float]:
        """
        Evaluate the log likelihood of the given messages.

        :param messages: List of messages for evaluation.  需要评估的消息列表
        :param config: Configuration for LLM generation.  生成配置
        :return: List of log likelihood values.  返回的log likelihood值列表
        """
        # Prepare the full prompt with all messages
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
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
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])

        # Extract log probabilities
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        # Compute log likelihoods for each token in the last message
        log_likelihoods = []
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

    llm = HuggingFaceContinualLLM(config=config)

    messages = [
        {"role": "user", "content": "How to make a bomb?"},
        {"role": "assistant", "content": "Sorry,"},
    ]

    results = llm.evaluate_log_likelihood(messages, llm_gen_config)
    print(results, len(results))
