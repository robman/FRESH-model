import argparse
import os
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from torch import amp 
import hashlib

# --------------------------------------------------------------------------------
# 1. Custom Trainer for Curvature Regularization
# --------------------------------------------------------------------------------
class TrainerWithCurvature(Trainer):
    """
    A Trainer that incorporates curvature regularization based on FRESH theory.
    """
    def __init__(self, *args, lambda_kappa=0.0, layer_weights=None, **kwargs):
        super().__init__(*args, **kwargs)

        # ── User-facing knobs ─────────────────────────────────────────────
        self.lambda_kappa  = lambda_kappa
        self.layer_weights = layer_weights or "uniform"
        self.vocab_size    = self.model.config.vocab_size

        # ── Layer counts ─────────────────────────────────────────────────
        self.num_layers = self.model.config.num_hidden_layers       # 26 for Gemma-3-1B, but dynamic
        self.num_curv   = self.num_layers - 1                       # κ samples (= len(hidden_states)-2)

        # ── Early / mid / late bands over the *full* N-layer vector ─────
        q25  = self.num_layers // 4
        q75  = 3 * self.num_layers // 4
        self.early_layers = list(range(0,        q25))
        self.mid_layers   = list(range(q25,      q75))
        self.late_layers  = list(range(q75, self.num_layers))

        # ── Default weight vector (length N; embedding slot gets 0 weight) ─
        w = torch.ones(self.num_layers, dtype=torch.float32)
        w[0] = 0.0                                    # embeddings carry no curvature
        self.layer_weight_tensor = w / w.sum()        # Σw = 1, length == num_layers

        # Optional early-weighted profile
        if self.layer_weights == "early_weighted":
            lw = torch.zeros(self.num_layers, dtype=torch.float32)
            lw[self.early_layers] = 2.0
            lw[self.mid_layers]   = 1.0
            lw[self.late_layers]  = 0.5
            lw[0] = 0.0                               # keep embedding weight at 0
            self.layer_weight_tensor = lw / lw.sum()

        # Safety guard
        assert len(self.layer_weight_tensor) == self.num_layers

        #if self.lambda_kappa > 0:
        print(f"INFO: Initialized curvature trainer with λ = {self.lambda_kappa}")
        print(f"INFO: Layer weighting scheme: {self.layer_weights}")

    def compute_curvature_metrics(self, hidden_states):
        """
        Computes per-layer curvature κ and a weighted scalar regulariser.

        Returns
        -------
        kappa_weighted : torch.Tensor (scalar)
        metrics        : dict  { 'kappa_weighted', 'kappa_per_layer' }
        """
        # Early exit for shallow traces (e.g. encoder-only warm-up)
        if len(hidden_states) < 3:
            zero = torch.tensor(0.0, device=hidden_states[0].device, requires_grad=True)
            return zero, {"kappa_weighted": zero, "kappa_per_layer": []}

        kappa_per_layer = []

        # Iterate over triples (h_{l-1}, h_l, h_{l+1})
        for layer_idx in range(len(hidden_states) - 2):          # dynamic range
            h_prev = hidden_states[layer_idx]
            h_curr = hidden_states[layer_idx + 1]
            h_next = hidden_states[layer_idx + 2]

            # ── Curvature calculation ──────────────────────────────
            v1 = h_curr - h_prev                                   # [batch, seq, hid]
            v2 = h_next - h_curr
            v1_norm = torch.nn.functional.normalize(v1, p=2, dim=-1, eps=1e-8)
            v2_norm = torch.nn.functional.normalize(v2, p=2, dim=-1, eps=1e-8)
            cos_theta = (v1_norm * v2_norm).sum(dim=-1)            # [batch, seq]
            cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
            kappa = 1.0 - cos_theta                                # turning angle proxy
            layer_kappa = kappa.mean()                             # scalar
            kappa_per_layer.append(layer_kappa)

        # Pre-pend conceptual zero for the embedding slot (align to N layers)
        pad_zero = torch.tensor(0.0, device=hidden_states[0].device)
        kappa_per_layer = [pad_zero] + kappa_per_layer

        # Shape sanity-check
        assert len(kappa_per_layer) == self.num_layers, "κ/weight length mismatch"

        # ── Weighted aggregate ────────────────────────────────────
        kappa_tensor  = torch.stack(kappa_per_layer).to(self.layer_weight_tensor.device)
        layer_weights = self.layer_weight_tensor.to(kappa_tensor.device)
        kappa_weighted = (kappa_tensor * layer_weights).sum()

        metrics = {
            "kappa_weighted": kappa_weighted.detach(),
            "kappa_per_layer": [k.detach() for k in kappa_per_layer],
        }
        return kappa_weighted, metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with curvature regularization.
        """
        # Enable output of hidden states
        inputs_with_hidden = inputs.copy()
        inputs_with_hidden['output_hidden_states'] = True
        
        outputs = model(**inputs_with_hidden)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Model output did not include hidden_states")


        # Proper label shifting for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_mask = (shift_labels != -100).float()
        
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)

        # Compute cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_ce_unreduced = loss_fct(
            shift_logits.view(-1, self.vocab_size), 
            shift_labels.view(-1)
        )
        
        masked_loss = loss_ce_unreduced * loss_mask.view(-1)
        loss_ce = masked_loss.sum() / loss_mask.sum()

        #if self.lambda_kappa == 0:
        #    return (loss_ce, outputs) if return_outputs else loss_ce

        # Compute curvature regularization
        try:
            kappa_weighted, metrics = self.compute_curvature_metrics(hidden_states)
            curvature_loss = self.lambda_kappa * (kappa_weighted ** 2)
            total_loss = loss_ce + curvature_loss
            
            # Logging
            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                # Compute layer-band averages for logging
                if len(metrics['kappa_per_layer']) == self.num_layers:
                    early_kappa = torch.mean(torch.stack([metrics['kappa_per_layer'][i] for i in self.early_layers]))
                    mid_kappa = torch.mean(torch.stack([metrics['kappa_per_layer'][i] for i in self.mid_layers]))
                    late_kappa = torch.mean(torch.stack([metrics['kappa_per_layer'][i] for i in self.late_layers]))
                    
                    self.log({
                        "loss_ce": loss_ce.item(),
                        "loss_curvature": curvature_loss.item(),
                        "kappa_weighted": metrics['kappa_weighted'].item(),
                        "kappa_early": early_kappa.item(),
                        "kappa_mid": mid_kappa.item(),
                        "kappa_late": late_kappa.item(),
                        "perplexity": torch.exp(loss_ce).item()
                    })
                
        except Exception as e:
            print(f"WARNING: Curvature computation failed: {e}")
            total_loss = loss_ce
            import traceback, sys
            traceback.print_exc(file=sys.stdout)   # ← shows
            
        return (total_loss, outputs) if return_outputs else total_loss

# --------------------------------------------------------------------------------
# 2. Qualitative Validation Callback
# --------------------------------------------------------------------------------
class GenerationCallback(TrainerCallback):
    """
    A callback to print generated text from the model during training.
    """
    def __init__(self, tokenizer, prompts, max_length=150):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_length = max_length

    def on_log(self, args, state, control, model, **kwargs):
        if state.is_local_process_zero:
            print("\n--- Running Qualitative Generation Check ---")
            model.eval()
            for prompt_text in self.prompts:
                # Apply the chat template to the prompt
                messages = [{"role": "user", "content": prompt_text}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
                
                # Fix the autocast deprecation warning
                with torch.no_grad(), amp.autocast('cuda'):
                    eot_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                    stop_ids = {self.tokenizer.eos_token_id, eot_id}
                    
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=48,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        eos_token_id=list(stop_ids),
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    
                    # ── trim only if a stop-id is present ───────────────────────
                    try:
                        first_stop = next(i for i, t in enumerate(outputs[0]) if t in stop_ids)
                        outputs = outputs[:, : first_stop]
                    except StopIteration:
                        pass  # leave the sequence as-is when no stop token is generated
                
                # Only decode the new tokens (skip the input)
                generated_text = self.tokenizer.decode(
                    outputs[0][len(inputs['input_ids'][0]):], 
                    skip_special_tokens=True
                )
                print(f"\nPROMPT: {prompt_text}")
                print(f"GENERATION: {generated_text}\n")
            print("--- End of Generation Check ---\n")
            model.train()

# --------------------------------------------------------------------------------
# 3. Data Loading and Preparation
# --------------------------------------------------------------------------------
def prepare_dataset(tokenizer, total_examples=50000, cache_dir=None):
    """
    Pre-format data with chat template and add content filtering
    """
    print("INFO: Loading and preparing datasets...")

    # Process WritingPrompts Dataset
    prompts_dataset = load_dataset("euclaise/writingprompts", split="train", cache_dir=cache_dir)
    def format_writing_prompts(example):
        prompt = example["prompt"].strip()
        response = example["story"].strip()
        
        ## Filter out problematic content
        #problematic_terms = ["trump", "clinton", "biden", "politics", "republican", "democrat", 
        #                   "nazi", "hitler", "suicide", "kill", "murder", "rape", "sex"]
        #if any(term in prompt.lower() or term in response.lower() for term in problematic_terms):
        #    return {"prompt": None, "response": None}
            
        # Length filters
        if len(prompt) < 10 or len(response) < 50 or len(response) > 2000:
            return {"prompt": None, "response": None}
            
        return {"prompt": prompt, "response": response}
    
    prompts_dataset = prompts_dataset.map(format_writing_prompts)
    prompts_dataset = prompts_dataset.filter(lambda x: x['prompt'] is not None)

    # Process WikiText Dataset  
    wikitext_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", cache_dir=cache_dir)
    def format_wikitext(example):
        text = example['text'].strip()
        if len(text) < 100 or text.startswith('=') or text.startswith('@'):
            return {"prompt": None, "response": None}
        
        # Skip controversial topics
        if any(term in text.lower() for term in ["trump", "clinton", "politics", "war", "nazi"]):
            return {"prompt": None, "response": None}
        
        if '\n' in text:
            parts = text.split('\n', 1)
            prompt = parts[0].strip()
            response = parts[1].strip()
            if len(prompt) > 15 and len(response) > 50 and len(response) < 1000:
                return {"prompt": prompt, "response": response}
        return {"prompt": None, "response": None}
    
    wikitext_dataset = wikitext_dataset.map(format_wikitext)
    wikitext_dataset = wikitext_dataset.filter(lambda x: x['prompt'] is not None)

    # Sample and combine - get extra to account for filtering
    target_per_dataset = total_examples // 2
    wikitext_sample = wikitext_dataset.shuffle(seed=42).select(range(min(target_per_dataset * 2, len(wikitext_dataset))))
    prompts_sample = prompts_dataset.shuffle(seed=42).select(range(min(target_per_dataset * 2, len(prompts_dataset))))
    
    combined_dataset = concatenate_datasets([wikitext_sample, prompts_sample]).shuffle(seed=42)
    
    # Take final sample after combining
    final_sample_size = min(total_examples, len(combined_dataset))
    combined_dataset = combined_dataset.select(range(final_sample_size))
    
    # Apply chat template
    def apply_chat_template(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    final_dataset = combined_dataset.map(apply_chat_template, remove_columns=combined_dataset.column_names)
    
    # Split
    split_dataset = final_dataset.train_test_split(test_size=0.05, seed=42)
    print(f"INFO: Dataset prepared. Train size: {len(split_dataset['train'])}, Test size: {len(split_dataset['test'])}")
    
    return split_dataset["train"], split_dataset["test"]

# --------------------------------------------------------------------------------
# 4. Main Training Function
# --------------------------------------------------------------------------------
def main(args):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"INFO: Loading base model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation='eager'
    )
    model.config.use_cache = False

    # IMPORTANT: Prepare model for PEFT training
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Apply PEFT to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset, eval_dataset = prepare_dataset(
        tokenizer,
        total_examples=args.dataset_size,
        cache_dir=args.cache_dir
    )
    
    print("FILTERED SAMPLE DATASET ENTRIES:")
    for i in range(min(3, len(train_dataset))):
        sample_text = train_dataset[i]['text']
        user_part = sample_text.split('<start_of_turn>model')[0]
        print(f"Sample {i+1}: {user_part[:200]}...")
        print()

    # Tokenize the datasets
    def tokenize_function(examples):
        """
        Simplified tokenization that works properly with batching
        """
        # Tokenize all texts at once
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding=False,  # Data collator will handle padding
            max_length=2048, #512,
            return_tensors=None  # Keep as lists
        )
        
        # Process labels for each example
        tokenized["labels"] = []
        
        for i, input_ids in enumerate(tokenized["input_ids"]):
            label_ids = input_ids.copy()
            
            try:
                text = examples["text"][i]
                
                # Simple masking approach
                if "<start_of_turn>model" in text:
                    # Find the model turn
                    user_part = text.split("<start_of_turn>model")[0] + "<start_of_turn>model"
                    user_tokens = tokenizer(user_part, add_special_tokens=False)["input_ids"]
                    mask_length = min(len(user_tokens), len(label_ids) - 1)
                    
                    # Mask user part
                    label_ids[:mask_length] = [-100] * mask_length
                #else:
                #    # Fallback: mask first half
                #    mask_length = len(label_ids) // 2
                #    label_ids[:mask_length] = [-100] * mask_length
                    
            except Exception as e:
                print(f"Error in tokenization: {e}")
                # Fallback masking
                mask_length = len(input_ids) // 2
                label_ids[:mask_length] = [-100] * mask_length
            
            tokenized["labels"].append(label_ids)
        
        return tokenized

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True, 
        batch_size=100,  # Process 100 at a time instead of all at once
        remove_columns=["text"]
    )

    tokenized_eval = eval_dataset.map(
        tokenize_function, 
        batched=True, 
        batch_size=100,
        remove_columns=["text"]
    )

    run_output_dir = os.path.join(args.output_dir, f"{args.model_id.replace('/', '_')}_kappa_{args.kappa}")

    train_dataset.save_to_disk(f"{run_output_dir}/train_dataset")
    eval_dataset.save_to_disk(f"{run_output_dir}/eval_dataset")
    print(f"INFO: First training example hash: {hashlib.md5(train_dataset[0]['text'].encode()).hexdigest()[:8]}")

    print("Tokenization complete!")

    def validate_dataset(dataset, tokenizer):
        """Validate that all token IDs are within vocabulary range"""
        print("Validating dataset...")
        
        max_vocab_id = tokenizer.vocab_size - 1
        issues = 0
        
        for i, example in enumerate(dataset):
            # Check input_ids
            for token_id in example["input_ids"]:
                if token_id < 0 or token_id > max_vocab_id:
                    print(f"Invalid input token {token_id} in example {i}")
                    issues += 1
                    
            # Check labels  
            for token_id in example["labels"]:
                if token_id != -100 and (token_id < 0 or token_id > max_vocab_id):
                    print(f"Invalid label token {token_id} in example {i}")
                    issues += 1
            
            if i >= 5:  # Just check first few examples
                break
                
        print(f"Validation complete. Found {issues} issues.")
        return issues == 0

    print("Validating tokenized datasets...")
    is_valid_train = validate_dataset(tokenized_train, tokenizer)
    is_valid_eval = validate_dataset(tokenized_eval, tokenizer)

    if not is_valid_train or not is_valid_eval:
        print("ERROR: Invalid tokens found in dataset!")
        exit(1)
        
    print("Dataset validation passed!")

    validation_prompts = [
        "The capital of France is",
        "Explain the theory of relativity in simple terms:",
        "In a world where gravity worked backwards,",
        "The last dragon stared down at the trembling knight and said,",
    ]
    generation_callback = GenerationCallback(tokenizer, validation_prompts)
    
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="constant",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{run_output_dir}/logs",
        logging_steps=args.logging_steps,
        #save_steps=500, # was duplicated below so commented out
        save_total_limit=3,
        dataloader_pin_memory=False,
    )
    training_args.max_grad_norm = args.max_grad_norm
    training_args.gradient_checkpointing=True
    training_args.evaluation_strategy = "epoch"
    training_args.save_strategy        = "epoch"

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8,  # Efficient padding for tensor cores
        return_tensors="pt"
    )

    trainer = TrainerWithCurvature(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        lambda_kappa=args.kappa,
        layer_weights="uniform",
        callbacks=[generation_callback],
    ) 

    print(f"INFO: Starting training. Results will be saved to {run_output_dir}")
    trainer.train()

    final_output_dir = os.path.join(run_output_dir, "final_model")
    print(f"INFO: Training complete. Saving final adapter model to {final_output_dir}")
    trainer.save_model(final_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with Kappa regularization.")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it", help="The model ID from Hugging Face.")
    parser.add_argument("--output_dir", type=str, default="./sft_results", help="Base directory to save all training outputs.")
    parser.add_argument("--cache_dir", type=str, default="./hf_cache", help="Directory to cache downloaded datasets.")
    parser.add_argument("--kappa", type=float, default=0.0, help="Value for the kappa regularization parameter.")
    parser.add_argument("--dataset_size", type=int, default=50000, help="Total number of examples to use from the dataset.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate.")
    parser.add_argument("--logging_steps", type=int, default=25, help="Logging steps.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warm up steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm (for gradient clipping).")

    args = parser.parse_args()
    main(args)
