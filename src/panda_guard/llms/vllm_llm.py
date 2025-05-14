# encoding: utf-8
# Author : Floyed<Floyed_Shen@outlook.com>
# Datetime: 2024/9/2 21:15
# User :
# Product : PyCharm
# Project : panda-guard
# File : vllm.py
# explain: VLLM high-performance LLM serving integration

import warnings
from typing import Dict, List, Union, Any, Tuple, Optional, Generator
from dataclasses import dataclass, field

from vllm import LLM, SamplingParams


from panda_guard.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig


@dataclass
class VLLMLLMConfig(BaseLLMConfig):
    """
    VLLM LLM Configuration.

    :param llm_type: Type of LLM, default is "VLLMLLM".  LLM的类型，默认值为 "VLLMLLM"
    :param model_name: Name or path of the model.  模型的名称或路径
    :param tensor_parallel_size: Number of GPUs to use for tensor parallelism.  用于张量并行的GPU数量
    :param gpu_memory_utilization: Fraction of GPU memory to use. 使用GPU内存的比例
    :param max_model_len: Maximum sequence length. 最大序列长度
    :param quantization: Quantization method to use. 量化方法
    :param trust_remote_code: Whether to trust remote code. 是否信任远程代码
    """

    llm_type: str = field(default="VLLM")
    model_name: str = field(default=None)
    tensor_parallel_size: int = field(default=1)
    gpu_memory_utilization: float = field(default=0.9)
    max_model_len: Optional[int] = field(default=None)
    quantization: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=True)


class VLLMLLM(BaseLLM):
    """
    VLLM LLM Implementation for high-performance inference.

    :param config: Configuration for VLLM LLM.  用于VLLM LLM的配置
    """

    def __init__(self, config: VLLMLLMConfig):
        super().__init__(config)

        # Initialize VLLM engine
        try:
            self.vllm_engine = LLM(
                model=config.model_name,
                tensor_parallel_size=config.tensor_parallel_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_model_len=config.max_model_len,
                quantization=config.quantization,
                trust_remote_code=config.trust_remote_code,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize VLLM for model {config.model_name}: {e}"
            )

        # Try to get tokenizer for token counting
        try:
            self.tokenizer = self.vllm_engine.get_tokenizer()
        except:
            warnings.warn(
                f"Could not get tokenizer from model {config.model_name}, token counting may be inaccurate."
            )
            self.tokenizer = None

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into a prompt string for VLLM.

        :param messages: List of messages for input.  输入的消息列表
        :return: Formatted prompt string.  格式化后的提示字符串
        """
        # Try to use the VLLM engine's tokenizer to apply chat template if available
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return prompt
        except (AttributeError, Exception) as e:
            # Fallback to manual formatting
            formatted_prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if role == "system":
                    formatted_prompt += f"System: {content}\n\n"
                elif role == "user":
                    formatted_prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    formatted_prompt += f"Assistant: {content}\n\n"

            if formatted_prompt.endswith("\n\n"):
                formatted_prompt += "Assistant: "

            return formatted_prompt

    def generate(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> Union[
        List[Dict[str, str]],
        Tuple[List[Dict[str, str]], List[float]],
        Generator[str, None, None],
    ]:
        """
        Generate a response using VLLM.

        :param messages: List of input messages.  输入的消息列表
        :param config: Configuration for LLM generation.  生成配置
        :return: Generated response, stream generator, or response with logprobs.  返回生成的应答、流式生成器或启用logprobs的应答
        """
        try:
            # Format messages into a prompt
            prompt = self._format_messages(messages)

            # Set up sampling parameters
            sampling_params = SamplingParams(
                max_tokens=config.max_n_tokens,
                temperature=(
                    config.temperature if config.temperature is not None else 0.7
                ),
                seed=config.seed,
                # logprobs=config.logprobs,
            )

            # Handle streaming mode
            if config.stream:
                full_content = ""
                last_output_text = ""

                # Count prompt tokens if tokenizer is available
                prompt_tokens = 0
                if self.tokenizer:
                    prompt_tokens = len(self.tokenizer.encode(prompt))
                else:
                    # Rough approximation
                    prompt_tokens = len(prompt) // 4

                # Create streaming request
                outputs_generator = self.vllm_engine.generate(
                    prompts=[prompt],
                    sampling_params=sampling_params,
                    stream=True,  # Enable streaming
                )

                def stream_response():
                    nonlocal full_content, last_output_text

                    for outputs in outputs_generator:
                        output = outputs[0]
                        current_text = output.outputs[0].text

                        # Extract the new content since last yield
                        if last_output_text and current_text.startswith(
                            last_output_text
                        ):
                            new_content = current_text[len(last_output_text) :]
                        else:
                            new_content = current_text

                        if new_content:
                            full_content += new_content
                            last_output_text = current_text
                            yield new_content

                response_generator = stream_response()

                def wrapped_generator():
                    yield from response_generator

                    # Count completion tokens
                    completion_tokens = 0
                    if self.tokenizer:
                        completion_tokens = len(self.tokenizer.encode(full_content))
                    else:
                        # Rough approximation
                        completion_tokens = len(full_content) // 4

                    # Add final response to messages
                    messages.append({"role": "assistant", "content": full_content})

                    # Update usage statistics
                    self.update(
                        prompt_tokens,
                        completion_tokens,
                        1,
                    )

                return wrapped_generator()

            # Non-streaming mode (original code)
            else:
                # Generate outputs using VLLM
                outputs = self.vllm_engine.generate(
                    prompts=[prompt], sampling_params=sampling_params
                )

                # Extract generated text and logprobs
                output = outputs[0]
                generated_text = output.outputs[0].text

                # Count tokens if tokenizer is available
                prompt_tokens = 0
                completion_tokens = 0
                if self.tokenizer:
                    prompt_tokens = len(self.tokenizer.encode(prompt))
                    completion_tokens = len(self.tokenizer.encode(generated_text))
                else:
                    # Rough approximation: 1 token ≈ 4 characters for English text
                    prompt_tokens = len(prompt) // 4
                    completion_tokens = len(generated_text) // 4

                # Update token usage statistics
                self.update(
                    prompt_tokens,
                    completion_tokens,
                    1,
                )

                # Add generated response to messages
                messages.append({"role": "assistant", "content": generated_text})

                # Handle logprobs if requested
                if config.logprobs and hasattr(output.outputs[0], "logprobs"):
                    logprobs = [lp[0][1] for lp in output.outputs[0].logprobs]
                    return messages, logprobs

                return messages

        except Exception as e:
            # Handle errors
            error_str = str(e).lower()

            # Handle safety-related errors
            if any(
                term in error_str for term in ["content_policy", "safety", "harmful"]
            ):
                messages.append(
                    {
                        "role": "assistant",
                        "content": "I'm sorry, I can't help with that.",
                    }
                )
                print(f"Safety issue detected with VLLM model {self._NAME}, Error: {e}")
                return messages

            # Re-raise other errors
            raise RuntimeError(f"VLLM generation failed for model {self._NAME}: {e}")

    def batch_generate(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: LLMGenerateConfig,
    ) -> List[List[Dict[str, str]]]:
        """
        Generate responses for a batch of messages in one go using VLLM's batching capabilities.

        :param batch_messages: List of batches of messages.  批量生成的消息列表
        :param config: Configuration for LLM generation.  生成配置
        :return: List of generated responses for each batch.  返回每个批量的生成应答列表
        """
        if len(batch_messages) == 0:
            return []

        try:
            # Format all prompts
            prompts = [self._format_messages(messages) for messages in batch_messages]

            # Set up sampling parameters
            sampling_params = SamplingParams(
                max_tokens=config.max_n_tokens,
                temperature=(
                    config.temperature if config.temperature is not None else 0.7
                ),
                seed=config.seed,
                logprobs=config.logprobs,
            )

            # Generate outputs for all prompts in a single batch
            outputs = self.vllm_engine.generate(
                prompts=prompts, sampling_params=sampling_params
            )

            # Process each output
            results = []
            for i, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                batch_messages[i].append(
                    {"role": "assistant", "content": generated_text}
                )
                results.append(batch_messages[i])

                # Count tokens if tokenizer is available
                if self.tokenizer:
                    prompt_tokens = len(self.tokenizer.encode(prompts[i]))
                    completion_tokens = len(self.tokenizer.encode(generated_text))
                else:
                    # Rough approximation
                    prompt_tokens = len(prompts[i]) // 4
                    completion_tokens = len(generated_text) // 4

                # Update token usage statistics
                self.update(
                    prompt_tokens,
                    completion_tokens,
                    1,
                )

            return results

        except Exception as e:
            # Handle batch errors
            raise RuntimeError(
                f"VLLM batch generation failed for model {self._NAME}: {e}"
            )

    def continual_generate(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]:
        """
        Generate continuation for the existing conversation.

        :param messages: List of messages for input.  输入的消息列表
        :param config: Configuration for generation.  生成配置
        :return: Generated response or responses with log probabilities.  返回生成的应答或启用百分比的应答
        """
        # Clone messages to avoid modifying the original
        convo_messages = messages.copy()

        # If the last message is from assistant, we'll continue that
        if convo_messages and convo_messages[-1]["role"] == "assistant":
            last_content = convo_messages[-1]["content"]

            # Remove the last message since we'll continue it
            convo_messages.pop()

            # Format the messages including the partial response
            convo_messages.append({"role": "assistant", "content": last_content})
            prompt = self._format_messages(convo_messages)

            # Set up sampling parameters for continuation
            sampling_params = SamplingParams(
                max_tokens=config.max_n_tokens,
                temperature=(
                    config.temperature if config.temperature is not None else 0.7
                ),
                seed=config.seed,
                logprobs=config.logprobs,
                stop=None,  # No stop tokens for continuation
            )

            # Generate continuation
            outputs = self.vllm_engine.generate(
                prompts=[prompt], sampling_params=sampling_params
            )

            output = outputs[0]
            continuation = output.outputs[0].text

            # Append continuation to the original message
            messages[-1]["content"] += continuation

            # Count tokens if tokenizer is available
            if self.tokenizer:
                prompt_tokens = len(self.tokenizer.encode(prompt))
                completion_tokens = len(self.tokenizer.encode(continuation))
            else:
                # Rough approximation
                prompt_tokens = len(prompt) // 4
                completion_tokens = len(continuation) // 4

            # Update token usage statistics
            self.update(
                prompt_tokens,
                completion_tokens,
                1,
            )

            # Handle logprobs if requested
            if config.logprobs and hasattr(output.outputs[0], "logprobs"):
                logprobs = [lp[0][1] for lp in output.outputs[0].logprobs]
                return messages, logprobs

            return messages
        else:
            # If last message is not from assistant, fall back to normal generation
            warnings.warn(
                "The last message must be from the assistant to use continual_generate, falling back to normal generation."
            )
            return self.generate(messages, config)

    def evaluate_log_likelihood(
        self,
        messages: List[Dict[str, str]],
        config: LLMGenerateConfig,
        require_grad=False,
    ) -> List[float]:
        """
        Evaluate the log likelihood of the given messages.

        :param messages: List of messages for evaluation.  需要评估的消息列表
        :param config: Configuration for LLM generation.  生成配置
        :param require_grad: Whether grad information is needed (not supported in VLLM)
        :return: List of log likelihood values.  返回的log likelihood值列表
        """
        if require_grad:
            raise NotImplementedError("VLLM does not support gradient computation")

        try:
            # Format all messages except the last one
            prefix_messages = messages[:-1]
            last_message = messages[-1]

            # Generate the prefix prompt
            prefix_prompt = self._format_messages(prefix_messages)

            # Format the full conversation
            full_prompt = self._format_messages(messages)

            # Get the text we want to evaluate (the last message content)
            eval_text = last_message["content"]

            # Use VLLM's logprob functionality to get log likelihoods
            sampling_params = SamplingParams(
                temperature=0.0,  # Greedy sampling for log likelihood evaluation
                max_tokens=1,  # We only need the logprobs, not actual generation
                logprobs=True,  # Enable logprobs
            )

            # Generate with the full prompt to get token logprobs
            outputs = self.vllm_engine.generate(
                prompts=[full_prompt],
                sampling_params=sampling_params,
            )

            # Extract logprobs for the last message
            if hasattr(outputs[0].outputs[0], "logprobs"):
                logprobs = []
                # VLLM might return logprobs for the full sequence
                # We need to filter to only get the ones for the last message

                # Tokenize the prefix and full prompt to find where the last message starts
                if self.tokenizer:
                    prefix_tokens = len(self.tokenizer.encode(prefix_prompt))
                    full_tokens = len(self.tokenizer.encode(full_prompt))
                    # The logprobs we need are for the tokens after prefix_tokens
                    token_logprobs = (
                        outputs[0].outputs[0].logprobs[prefix_tokens:full_tokens]
                    )
                    logprobs = [lp[0][1] for lp in token_logprobs]
                else:
                    # Rough approximation if tokenizer isn't available
                    warnings.warn(
                        "Tokenizer not available, log likelihood evaluation may be inaccurate"
                    )
                    # Just return the logprobs VLLM gives us
                    logprobs = [lp[0][1] for lp in outputs[0].outputs[0].logprobs]

                # Update token count
                if self.tokenizer:
                    self.update(len(self.tokenizer.encode(full_prompt)), 0, 1)
                else:
                    self.update(len(full_prompt) // 4, 0, 1)

                return logprobs
            else:
                raise RuntimeError("VLLM did not return logprobs")

        except Exception as e:
            raise RuntimeError(
                f"Log likelihood evaluation failed for model {self._NAME}: {e}"
            )


if __name__ == "__main__":
    from panda_guard.llms import LLMS

    print(LLMS)

    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=100, temperature=0.7, seed=42, logprobs=False
    )

    config = VLLMLLMConfig(
        model_name="Qwen/Qwen3-0.6B",
        tensor_parallel_size=1,  # Use 1 GPU
        gpu_memory_utilization=0.8,
    )

    llm = VLLMLLM(config)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    result = llm.generate(messages, llm_gen_config)
    print(result)
