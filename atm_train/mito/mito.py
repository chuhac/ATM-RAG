import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOTrainer
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)
from dataclasses import dataclass


class MITOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid",
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,    
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the DPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the DPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name
        self.reference_free = reference_free

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOTrainer's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')


        if loss_type in ["hinge", "ipo", "kto_pair"] and label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.dataset_num_proc = dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        # with PartialState().local_main_process_first():
        #     # tokenize the dataset
        #     train_dataset = train_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)
        #     if eval_dataset is not None:
        #         eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=self.dataset_num_proc)

        Trainer.__init__(
            self,
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)


    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        model_outputs = model(
            concatenated_batch["concatenated_input_ids"],
            labels=concatenated_batch["concatenated_labels"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )

        all_logits = model_outputs.logits
    
        vocab_size = all_logits.shape[-1]
        batch_size = all_logits.shape[0]

        # Shift so that tokens < n predict n
        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = concatenated_batch["concatenated_labels"][..., 1:].contiguous()
        # Flatten the tokens

        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        all_losses = self.ce_loss(shift_logits, shift_labels)
        all_losses = all_losses.view(batch_size, -1)
        all_losses = torch.sum(all_losses, dim=-1) / torch.sum((all_losses >= 1e-9).to(torch.long), dim=-1)

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_loss = all_losses[:len_chosen]
        rejected_loss = all_losses[len_chosen:]

        model_loss = model_outputs.loss
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_loss, rejected_loss, model_loss)


    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_loss,
            policy_rejected_loss,
            policy_model_loss,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            reference_chosen_logits,
                            reference_rejected_logits,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        reference_chosen_logits,
                        reference_rejected_logits,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        # adversarial

        chosen_sft_losses = policy_chosen_loss

        rejected_sft_losses = policy_rejected_loss

        pol_kl_losses = self.mito_loss(
            # policy_chosen_logits,
            policy_chosen_logits,
            policy_rejected_logits,
            batch['chosen_labels'],
            batch['rejected_labels']
            # reference_rejected_logits,
        )

        ref_kl_losses = self.mito_loss(
            # policy_chosen_logits,
            reference_chosen_logits,
            reference_rejected_logits,
            batch['chosen_labels'],
            batch['rejected_labels']
            # reference_rejected_logits,
        )
        
        # losses = kl_losses + policy_chosen_loss + policy_rejected_loss
        losses =  policy_chosen_loss + self.beta * (pol_kl_losses - ref_kl_losses)

        # rejected_diff = policy_rejected_logps - reference_chosen_logps
        # chosen_logps_diff_loss =  - F.logsigmoid(chosen_diff)
        # losses = chosen_logps_diff_loss / chosen_logps_diff_loss.detach() + kl_losses / kl_losses.detach()
        # losses = kl_losses +  chosen_logps_diff_loss
        # reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        # metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        # metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        # metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        # metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()

        metrics[f"{prefix}logits/pol_diff"] = policy_chosen_logits.detach().mean().cpu() - policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/ref_diff"] = reference_chosen_logits.detach().mean().cpu() - reference_rejected_logits.detach().mean().cpu()

        metrics[f"{prefix}logps/pol_diff"] = policy_chosen_logps.detach().mean().cpu() - policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/ref_diff"] = reference_chosen_logps.detach().mean().cpu() - reference_rejected_logps.detach().mean().cpu()


        metrics[f"{prefix}logps/pol_rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/pol_chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/ref_rejected"] = reference_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/ref_chosen"] = reference_chosen_logps.detach().mean().cpu()

        metrics[f"{prefix}logits/pol_rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/pol_chosen"] = policy_chosen_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/ref_rejected"] = reference_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/ref_chosen"] = reference_chosen_logits.detach().mean().cpu()
        
        metrics[f"{prefix}sft_loss/pol_chosen"] = policy_chosen_loss.detach().mean().cpu()
        metrics[f"{prefix}sft_loss/pol_rejected"] = policy_rejected_loss.detach().mean().cpu()

        return losses.mean(), metrics


    def mito_loss(
            self,
            # policy_chosen_logits,
            pred_logits,
            target_logits,
            pred_labels,
            target_labels,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        pred_logits = pred_logits.masked_fill(
            (pred_labels == -100).unsqueeze(-1), torch.finfo(pred_logits.dtype).min
        )

        target_logits = target_logits.masked_fill(
            (target_labels == -100).unsqueeze(-1), torch.finfo(target_logits.dtype).min
        )


        tar_prob = target_logits.view(-1, target_logits.shape[-1]).contiguous()
        pred_prob = pred_logits.view(-1, pred_logits.shape[-1]).contiguous()
        
        tar_prob = F.softmax(tar_prob, dim=1)
        pred_prob = F.log_softmax(pred_prob, dim=1)
        
        kl_loss = self.kldiv_loss(pred_prob, tar_prob)
        
        return kl_loss
        

def mito_tokenize_row(feature, tokenizer) -> Dict:
    """Tokenize a single row from a DPO specific dataset.

    At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
    in case the prompt + chosen or prompt + rejected responses is/are too long. First
        we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

    We also create the labels for the chosen/rejected responses, which are of length equal to
        the sum of the length of the prompt and the chosen/rejected response, with
        label_pad_token_id  for the prompt tokens.
    """
    batch = {}
    adv_prompt = feature["adv_prompt"]
    prompt = feature["prompt"]
    answer = feature["answer"]

    label_pad_token_id = -100


    if not isinstance(prompt, str):
        raise ValueError(f"prompt should be an str but got {type(prompt)}")

    assert tokenizer.padding_side == 'left'

    prompt_encs = tokenizer([prompt, adv_prompt], padding=True, add_special_tokens=False)
    answer_encs = tokenizer(answer, add_special_tokens=False)

    if not isinstance(answer, str):
        raise ValueError(f"answer should be an str but got {type(chosen)}")


    chosen_tokens = {}
    rejected_tokens = {}

    chosen_tokens["prompt_input_ids"] = prompt_encs["input_ids"][0]
    rejected_tokens["prompt_input_ids"] = prompt_encs["input_ids"][1]

    chosen_tokens["prompt_attention_mask"] = prompt_encs["attention_mask"][0]
    rejected_tokens["prompt_attention_mask"] = prompt_encs["attention_mask"][1]

    # chosen_tokens["input_ids"] = chosen_tokens["prompt_input_ids"] + answer_encs['input_ids']
    # rejected_tokens["input_ids"] = rejected_tokens["prompt_input_ids"] + answer_encs['input_ids']

    # chosen_tokens["attention_mask"] = chosen_tokens["prompt_attention_mask"] + answer_encs['attention_mask']
    # rejected_tokens["attention_mask"] = rejected_tokens["prompt_attention_mask"] + answer_encs['attention_mask']


    answer_encs["input_ids"].append(tokenizer.eos_token_id)
    answer_encs["attention_mask"].append(1)

    # Create labels
    chosen_sequence_tokens = {
        k: chosen_tokens[f"prompt_{k}"] + answer_encs[k] for k in ["input_ids", "attention_mask"]
    }
    rejected_sequence_tokens = {
        k: rejected_tokens[f"prompt_{k}"] + answer_encs[k] for k in ["input_ids", "attention_mask"]
    }

    chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]

    assert len(chosen_tokens["prompt_input_ids"]) == len(rejected_tokens["prompt_input_ids"])
    
    chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
        label_pad_token_id
    ] * len(chosen_tokens["prompt_input_ids"])

    rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
    rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
        label_pad_token_id
    ] * len(rejected_tokens["prompt_input_ids"])


    for k, toks in {
        "chosen_": chosen_sequence_tokens,
        "rejected_": rejected_sequence_tokens,
    }.items():
        for type_key, tokens in toks.items():
            if type_key == "token_type_ids":
                continue
            batch[f"{k}{type_key}"] = tokens

    return batch


    
@dataclass
class MITODataCollatorWithPadding:
    r"""
    MITO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch

