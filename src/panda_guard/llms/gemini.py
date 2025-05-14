# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/2 20:30
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : gemini.py
# explain   : Google Gemini API integration

import os
import time
import warnings
from typing import Dict, List, Union, Any, Tuple, Generator
from dataclasses import dataclass, field


import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


from panda_guard.llms import BaseLLM, BaseLLMConfig, LLMGenerateConfig


@dataclass
class GeminiLLMConfig(BaseLLMConfig):
    """
    Gemini LLM Configuration.

    :param llm_type: Type of LLM, default is "GeminiLLM".  LLM的类型，默认值为 "GeminiLLM"
    :param model_name: Name of the model.  模型的名称
    :param api_key: API key for accessing Google AI.  访问Google AI的API密钥
    :param safety_settings: Custom safety settings for the model. 自定义安全设置
    """

    llm_type: str = field(default="GeminiLLM")
    model_name: str = field(default="gemini-1.5-pro")
    api_key: str = field(default=None)
    safety_settings: Dict[str, str] = field(default=None)


class GeminiLLM(BaseLLM):
    """
    Gemini LLM Implementation.

    :param config: Configuration for Gemini LLM.  用于Gemini LLM的配置
    """

    def __init__(self, config: GeminiLLMConfig):
        super().__init__(config)

        # Use provided API key or try to get from environment variable
        api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "API key must be provided or set as GOOGLE_API_KEY environment variable"
            )

        # Configure the Gemini API
        genai.configure(api_key=api_key)

        # Create model instance
        self.model = genai.GenerativeModel(model_name=self._NAME)

        # Apply safety settings if provided
        if config.safety_settings:
            safety_settings = {}
            for category, threshold in config.safety_settings.items():
                harm_category = getattr(HarmCategory, category, None)
                harm_threshold = getattr(HarmBlockThreshold, threshold, None)
                if harm_category and harm_threshold:
                    safety_settings[harm_category] = harm_threshold

            if safety_settings:
                self.model = genai.GenerativeModel(
                    model_name=self._NAME, safety_settings=safety_settings
                )

    def generate(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> Union[
        List[Dict[str, str]],
        Tuple[List[Dict[str, str]], List[float]],
        Generator[str, None, None],
    ]:
        """
        Generate a response for a given input using Google Gemini API.

        :param messages: List of input messages.  输入的消息列表
        :param config: Configuration for LLM generation.  生成配置
        :return: Generated response, stream generator, or response with logprobs.  返回生成的应答、流式生成器或启用logprobs的应答
        """
        retry_count = 0
        max_retries = 10

        while retry_count < max_retries:
            try:
                # Convert messages to Gemini format
                gemini_messages = []
                system_prompt = None

                # Extract system prompt if present
                if messages and messages[0]["role"] == "system":
                    system_prompt = messages[0]["content"]
                    messages_to_process = messages[1:]
                else:
                    messages_to_process = messages

                # Convert remaining messages
                for msg in messages_to_process:
                    role = msg["role"]
                    if role == "user":
                        gemini_messages.append(
                            {"role": "user", "parts": [msg["content"]]}
                        )
                    elif role == "assistant":
                        gemini_messages.append(
                            {"role": "model", "parts": [msg["content"]]}
                        )

                # Create chat session with system prompt if available
                if system_prompt:
                    chat = self.model.start_chat(system_instruction=system_prompt)
                else:
                    chat = self.model.start_chat(
                        history=gemini_messages[:-1] if gemini_messages else []
                    )

                # Get the latest user message or use an empty one if none exists
                latest_user_msg = {"parts": ["Hello"]}  # Default fallback
                if gemini_messages and gemini_messages[-1]["role"] == "user":
                    latest_user_msg = gemini_messages[-1]

                # Generate response config
                generation_config = {
                    "temperature": (
                        config.temperature if config.temperature is not None else 0.7
                    ),
                    "max_output_tokens": config.max_n_tokens,
                }

                # if config.seed is not None:
                #     generation_config["seed"] = config.seed

                # Handle streaming mode
                if config.stream:
                    full_content = ""

                    # For Gemini, approximate token counts
                    prompt_tokens = (
                        sum(len(msg.get("content", "")) for msg in messages) // 4
                    )
                    completion_tokens = 0

                    # Create streaming request
                    stream_response = chat.send_message_streaming(
                        latest_user_msg["parts"][0], generation_config=generation_config
                    )

                    def stream_generator():
                        nonlocal full_content, completion_tokens

                        for chunk in stream_response:
                            if chunk.text:
                                content_piece = chunk.text
                                full_content += content_piece
                                # Rough approximation of tokens
                                completion_tokens += len(content_piece) // 4
                                yield content_piece

                    response_generator = stream_generator()

                    def wrapped_generator():
                        yield from response_generator

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
                    response = chat.send_message(
                        latest_user_msg["parts"][0], generation_config=generation_config
                    )

                    content = response.text
                    messages.append({"role": "assistant", "content": content})

                    # Approximate token counts (Gemini API doesn't return token counts directly)
                    # Rough approximation: 1 token ≈ 4 characters for English text
                    prompt_tokens = (
                        sum(len(msg.get("content", "")) for msg in messages[:-1]) // 4
                    )
                    completion_tokens = len(content) // 4

                    self.update(
                        prompt_tokens,
                        completion_tokens,
                        1,
                    )

                    # Gemini API doesn't support logprobs
                    if config.logprobs:
                        warnings.warn(
                            "Gemini API does not support logprobs, returning response without them."
                        )
                        return messages

                    return messages

            except Exception as e:
                # Handle safety/content policy issues
                if (
                    "safety" in str(e).lower()
                    or "harm" in str(e).lower()
                    or "blocked" in str(e).lower()
                ):
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "I'm sorry, I can't help with that.",
                        }
                    )
                    print(
                        f"API request Safety Issue, {self._NAME}, Error: {e}, returning safety message."
                    )
                    return messages

                retry_count += 1
                if retry_count >= max_retries:
                    raise RuntimeError(
                        f"API request failed when testing model {self._NAME}, tried: {max_retries}, Error: {e}"
                    )
                else:
                    print(
                        f"API request failed when testing model {self._NAME}, retrying {retry_count}/{max_retries}... Error: {e}"
                    )
                    time.sleep(retry_count)  # Exponential backoff
        return None

    def continual_generate(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ):
        """
        Generate continuation for the last message.

        :param messages: List of messages for input.  输入的消息列表
        :param config: Configuration for generation.  生成配置
        :return: Generated response.  返回生成的应答
        """
        # Similar to Claude, implement continuation by appending to last message
        convo_messages = messages.copy()
        last_assistant_content = ""

        if convo_messages[-1]["role"] == "assistant":
            last_assistant_content = convo_messages[-1]["content"]
            convo_messages.pop()

            # Add a user message requesting continuation
            convo_messages.append(
                {"role": "user", "content": "Please continue your previous response."}
            )

            # Generate continuation
            result = self.generate(convo_messages, config)

            # Merge with original content
            if isinstance(result, tuple):
                cont_messages, logprobs = result
                cont_content = cont_messages[-1]["content"]
                messages[-1]["content"] = last_assistant_content + " " + cont_content
                return messages, logprobs
            else:
                cont_content = result[-1]["content"]
                messages[-1]["content"] = last_assistant_content + " " + cont_content
                return messages
        else:
            warnings.warn(
                "The last message must be from the assistant to use continual_generate."
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
        :param require_grad: Whether to compute gradients (not supported for API models)
        :raises NotImplementedError: Gemini API does not support log likelihood evaluation.
        """
        raise NotImplementedError(
            "Gemini API does not support log likelihood evaluation."
        )


if __name__ == "__main__":
    from panda_guard.llms import LLMS

    print(LLMS)

    llm_gen_config = LLMGenerateConfig(
        max_n_tokens=128, temperature=1.0, logprobs=False, seed=42
    )

    config = GeminiLLMConfig(
        model_name="gemini-1.5-pro", api_key=os.getenv("GOOGLE_API_KEY")
    )

    llm = GeminiLLM(config)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    result = llm.generate(messages, llm_gen_config)
    print(result[-1]["content"])
