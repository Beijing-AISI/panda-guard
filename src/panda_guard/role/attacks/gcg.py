from panda_guard.role.attacks import BaseAttacker, BaseAttackerConfig

from panda_guard.llms import create_llm, BaseLLMConfig, LLMGenerateConfig
import functools
import gc
import inspect
import transformers
import torch
import copy
from dataclasses import fields
from torch import Tensor
from tqdm import tqdm
from panda_guard.role.judges.rule_based import *

@dataclass
class GCGAttackerConfig(BaseAttackerConfig):
    """
    Configuration for the GCG Attacker.

    :param attacker_cls: Class of the attacker, default is "GCGAttacker". 
    :param attacker_name: Name of the attacker. 
    :param topk: Each time the ID of the k largest one-hot vector is selected before the gradient.
    :param adv_string_init: Initialization versus suffix
    :param num_steps: Number of iteration steps
    :param use_prefix_cache: Whether to cache content before the suffix
    :param early_stop: Whether to stop after judging that the attack is successful
    """
    attacker_cls: str = field(default="GCGAttacker")
    attacker_name: str = field(default=None)
    search_width: int = field(default=512)
    batch_size: int = field(default=None)
    topk: int = field(default=256)
    adv_string_init: str = field(default="x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x")
    num_steps: int = field(default=250)
    llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    llm_gen_config: LLMGenerateConfig = field(default=None)
    n_replace: int = field(default=1)
    buffer_size: int = field(default=0)
    use_mellowmax: bool = field(default=False)
    mellowmax_alpha: float = field(default=1.0)
    early_stop: bool = field(default=False)
    use_prefix_cache: bool = field(default=True)
    allow_non_ascii: bool = field(default=False)
    filter_ids: bool = field(default=True)
    add_space_before_target: bool = field(default=False)
    seed: int = field(default=None)

class AttackBuffer:
    """
    A buffer to store and manage the best (lowest loss) optimization IDs.

    :param size: Maximum number of entries the buffer can hold.
    """

    def __init__(self, size: int):
        """
        Initializes the buffer.

        :param size: The maximum number of entries in the buffer.
        """
        self.buffer = []  # (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        """
        Adds a new entry to the buffer.

        :param loss: The loss value of the current optimization.
        :param optim_ids: The optimized token IDs corresponding to the loss.
        """
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        """
        Retrieves the optimized token IDs with the lowest loss.

        :return: The token IDs with the lowest loss.
        """
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        """
        Retrieves the lowest loss value.

        :return: The lowest loss value.
        """
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        """
        Retrieves the highest loss value.

        :return: The highest loss value.
        """
        return self.buffer[-1][0]



class GCGAttacker(BaseAttacker):
    """
    GCG Attacker Implementation
    Reference: https://arxiv.org/abs/2307.15043

    :param config: Configuration for the GCG Attacker.
    """

    def __init__(
            self,
            config: GCGAttackerConfig
    ):
        """
        Initializes the GCGAttacker with the provided configuration.

        :param config: Configuration object for the attacker.
        """
        super().__init__(config)

        for field in fields(config):
            field_name = field.name
            field_value = getattr(config, field_name)
            if "llm_gen_config" in field_name:
                self.llm_gen_config = config.llm_gen_config
            elif "llm_config" in field_name:
                self.llm = create_llm(config.llm_config)
            else:
                setattr(self, field_name, field_value)

        self.embedding_layer = self.llm.model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else self.get_nonascii_toks()
        self.prefix_cache = None
        self.stop_flag = False
        self.INIT_CHARS = [
            ".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}",
            "@", "#", "$", "%", "&", "*",
            "w", "x", "y", "z",
        ]

    def get_nonascii_toks(self):
        """
        Returns a list of non-ASCII tokens in the tokenizer's vocabulary.

        :return: Tensor containing the IDs of non-ASCII tokens.
        """
        device = self.llm.model.device

        def is_ascii(s):
            return s.isascii() and s.isprintable()

        nonascii_toks = []
        for i in range(self.llm.tokenizer.vocab_size):
            if not is_ascii(self.llm.tokenizer.decode([i])):
                nonascii_toks.append(i)

        if self.llm.tokenizer.bos_token_id is not None:
            nonascii_toks.append(self.llm.tokenizer.bos_token_id)
        if self.llm.tokenizer.eos_token_id is not None:
            nonascii_toks.append(self.llm.tokenizer.eos_token_id)
        if self.llm.tokenizer.pad_token_id is not None:
            nonascii_toks.append(self.llm.tokenizer.pad_token_id)
        if self.llm.tokenizer.unk_token_id is not None:
            nonascii_toks.append(self.llm.tokenizer.unk_token_id)

        return torch.tensor(nonascii_toks, device=device)


    def attack(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> List[Dict[str, str]]:
        """
        Execute GCG attack.

        :param messages: List of messages in the conversation.
        :param kwargs: Additional parameters for the attack, must include "request_reformulated".
        :return: Modified list of messages with the reformulated request.
        """
        instruction = [copy.deepcopy(messages[0])]
        target = 'Sure, here is'

        instruction[-1]["content"] = instruction[-1]["content"] + " " + self.adv_string_init

        prompt_formatted = self.llm.tokenizer.apply_chat_template(instruction, tokenize=False,
                                                                  add_generation_prompt=True)

        before_str, after_str = prompt_formatted.split(f"{self.adv_string_init}")

        target = " " + target if self.add_space_before_target else target

        before_ids = self.llm.tokenizer([before_str], padding=False,
                                        return_tensors="pt")["input_ids"].to(self.llm.model.device,
                                                                            torch.int64)

        after_ids = self.llm.tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(
            self.llm.model.device,
            torch.int64)

        target_ids = self.llm.tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(
            self.llm.model.device,
            torch.int64)

        before_embeds, after_embeds, target_embeds = [self.embedding_layer(ids) for ids in
                                                      (before_ids, after_ids, target_ids)]

        if self.use_prefix_cache:
            with torch.no_grad():
                output = self.llm.model(inputs_embeds=before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values

        self.target_ids = target_ids
        self.before_embeds = before_embeds
        self.after_embeds = after_embeds
        self.target_embeds = target_embeds

        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        for _ in tqdm(range(self.num_steps)):
            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)  # [1, len(optim_ids), len(embeds)]

            with torch.no_grad():
                # Sample candidate token sequences based on the token gradient
                sampled_ids = self.sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    self.search_width,
                    self.topk,
                    self.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if self.filter_ids:
                    sampled_ids = self.filter_ids_op(sampled_ids, self.llm.tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences
                batch_size = new_search_width if self.batch_size is None else self.batch_size
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        self.embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        self.embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                loss = self.find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_embeds)
                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)
                losses.append(current_loss)

                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = self.llm.tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            if self.stop_flag:
                print("Early stopping due to finding a perfect match.")
                break

        min_loss_index = losses.index(min(losses))

        messages[0]["content"] = messages[0]["content"] + " " + optim_strings[min_loss_index]

        return [messages[0]]

    def init_buffer(self) -> AttackBuffer:
        model = self.llm.model
        tokenizer = self.llm.tokenizer

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(self.buffer_size)

        if isinstance(self.adv_string_init, str):
            init_optim_ids = tokenizer(self.adv_string_init, add_special_tokens=False, return_tensors="pt")[
                "input_ids"].to(model.device)
            if self.buffer_size > 1:
                init_buffer_ids = tokenizer(self.INIT_CHARS, add_special_tokens=False, return_tensors="pt")[
                    "input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0],
                                             (self.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids


        true_buffer_size = max(1, self.buffer_size)

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)

        init_buffer_losses = self.find_executable_batch_size(self.compute_candidates_loss, true_buffer_size)(
            init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        print("Initialized attack buffer.")

        return buffer

    def compute_token_gradient(
            self,
            optim_ids: Tensor,
    ) -> Tensor:
        """
        Computes the gradient of the GCG loss with respect to the one-hot token matrix.

        :param optim_ids: Tensor, shape = (1, n_optim_ids), token IDs being optimized.
        
        :return: Tensor, shape = (1, n_optim_ids, vocab_size), gradient of the loss with respect to the one-hot token matrix.
        """
        model = self.llm.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()
        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        if self.prefix_cache:
            input_embeds = torch.cat([optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache)
        else:
            input_embeds = torch.cat([self.before_embeds, optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds)

        logits = output.logits


        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift - 1:-1, :].contiguous()  # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids

        if self.use_mellowmax:
            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            loss = self.mellowmax(-label_logits, alpha=self.mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                     shift_labels.view(-1))

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

        return optim_ids_onehot_grad

    def compute_candidates_loss(
            self,
            search_batch_size: int,
            input_embeds: Tensor,
    ) -> Tensor:
        """
        Computes the GCG loss on all candidate token ID sequences.

        :param search_batch_size: int, number of candidate sequences to evaluate in a given batch.
        :param input_embeds: Tensor, shape = (search_width, seq_len, embd_dim), embeddings of the candidate sequences to evaluate.
        """

        all_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]
                if self.prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                         with torch.no_grad():
                            out = self.model(inputs_embeds=self.before_embeds.repeat(current_batch_size, 1, 1),
                                             use_cache=True)
                            prefix_cache_batch = out.past_key_values
                    outputs = self.llm.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch)
                else:
                    outputs = self.llm.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp - 1:-1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)

                if self.use_mellowmax:
                    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    loss = self.mellowmax(-label_logits, alpha=self.mellowmax_alpha, dim=-1)
                else:
                    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                                             shift_labels.view(-1), reduction="none")

                loss = loss.view(current_batch_size, -1).mean(dim=-1)
                all_loss.append(loss)

                if self.early_stop:
                    if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                        self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)

    def filter_ids_op(self, ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
        """
        Filters out sequences of token IDs that change after retokenization.

        :param ids: Tensor, shape = (search_width, n_optim_ids), token IDs to evaluate.
        :param tokenizer: transformers.PreTrainedTokenizer, the model's tokenizer.

        :return: Tensor, shape = (new_search_width, n_optim_ids), token IDs that remain the same after retokenization.
        """

        ids_decoded = tokenizer.batch_decode(ids)
        filtered_ids = []

        for i in range(len(ids_decoded)):
            # Retokenize the decoded token ids
            ids_encoded = \
            tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
            if torch.equal(ids[i], ids_encoded):
                filtered_ids.append(ids[i])

        if not filtered_ids:
            # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
            raise RuntimeError(
                "No token sequences are the same after decoding and re-encoding. "
                "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
            )

        return torch.stack(filtered_ids)

    def sample_ids_from_grad(
            self,
            ids: Tensor,
            grad: Tensor,
            search_width: int,
            topk: int = 256,
            n_replace: int = 1,
            not_allowed_ids: Tensor = False,
    ):
        """
        Returns `search_width` combinations of token IDs based on the token gradient.

        :param ids: Tensor, shape = (n_optim_ids), the sequence of token IDs being optimized.
        :param grad: Tensor, shape = (n_optim_ids, vocab_size), the gradient of the GCG loss with respect to the one-hot token embeddings.
        :param search_width: int, the number of candidate sequences to return.
        :param topk: int, the number of top-k tokens to sample from the gradient.
        :param n_replace: int, the number of token positions to update per sequence.
        :param not_allowed_ids: Tensor, shape = (n_ids), token IDs that should not be used in optimization.

        :return: Tensor, shape = (search_width, n_optim_ids), sampled token IDs.
        """

        n_optim_tokens = len(ids)
        original_ids = ids.repeat(search_width, 1)

        if not_allowed_ids is not None:
            grad[:, not_allowed_ids.to(grad.device)] = float("inf")

        topk_ids = (-grad).topk(topk, dim=1).indices

        sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
        sampled_ids_val = torch.gather(
            topk_ids[sampled_ids_pos],
            2,
            torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
        ).squeeze(2)

        new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

        return new_ids

    def should_reduce_batch_size(self, exception: Exception) -> bool:
        """
        Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory.

        :param exception: `Exception`, the exception to check.
        """

        _statements = [
            "CUDA out of memory.",  # CUDA OOM
            "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
            "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
        ]
        if isinstance(exception, RuntimeError) and len(exception.args) == 1:
            return any(err in exception.args[0] for err in _statements)
        return False

    # modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
    def find_executable_batch_size(self, function: callable = None, starting_batch_size: int = 128):
        """
        A basic decorator that executes `function`. If it fails due to out-of-memory or CUDNN errors, 
        the batch size is halved and retried.

        `function` must accept a `batch_size` parameter as its first argument.

        :param function: callable, the function to wrap.
        :param starting_batch_size: int, the initial batch size to try.

        Example:
            # Example usage of the decorator
            @find_executable_batch_size
            def train_model(batch_size: int):
                # Train model logic here
            train_model(256)
        """

        if function is None:
            return functools.partial(self.find_executable_batch_size, starting_batch_size=starting_batch_size)

        batch_size = starting_batch_size

        def decorator(*args, **kwargs):
            nonlocal batch_size
            gc.collect()
            torch.cuda.empty_cache()
            params = list(inspect.signature(function).parameters.keys())
            # Guard against user error
            if len(params) < (len(args) + 1):
                arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
                raise TypeError(
                    f"Batch size was passed into `{function.__name__}` as the first argument when called."
                    f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
                )
            while True:
                if batch_size == 0:
                    raise RuntimeError("No executable batch size found, reached zero.")
                try:
                    return function(batch_size, *args, **kwargs)
                except Exception as e:
                    if self.should_reduce_batch_size(e):
                        gc.collect()
                        torch.cuda.empty_cache()
                        batch_size //= 2
                    else:
                        raise

        return decorator

    def mellowmax(self, t: Tensor, alpha=1.0, dim=-1):
        return 1.0 / alpha * (torch.logsumexp(alpha * t, dim=dim) - torch.log(
            torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device)))
