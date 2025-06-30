import argparse
import json
import logging
import os
import time
import uuid
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Global Configuration ---
# Setup logging to provide clear, informative output during the capture process.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# --- Global State ---
# These are global to be easily accessible and clearable within the hook functions and main loop.
current_prompt_run_activations = {} # Stores activations for the current prompt and run.
hook_handles = [] # To store hook handles for proper removal.

# --- Helper function ---
def append_to_csv(filepath: str, rows: list, expected_headers: list[str]):
    if not rows: return
    file_exists = os.path.isfile(filepath)
    is_empty = not file_exists or os.path.getsize(filepath) == 0
    df = pd.DataFrame(rows)
    for h in expected_headers:
        if h not in df.columns: df[h] = pd.NA
    df = df.reindex(columns=expected_headers)
    try:
        # Use quoting=csv.QUOTE_ALL to properly handle text with newlines/commas
        df.to_csv(filepath, mode='a', header=is_empty, index=False,
                 lineterminator='\n', quoting=1, escapechar='\\')
    except Exception as e:
        logger.error(f"Error writing to CSV {filepath}: {e}")
        error_csv_path = filepath.replace(".csv", f"_error_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv")
        try:
            df.to_csv(error_csv_path, mode='w', header=True, index=False,
                     lineterminator='\n', quoting=1, escapechar='\\')
        except Exception as e2: logger.error(f"Could not save data to fallback CSV {error_csv_path}: {e2}")

# --- Core Generation Function for CI03 ---
def generate_response(
        model,
        tokenizer,
        inputs,               # tokenised prompt (batch=1) ─ already on device
        max_new_tokens,
        stop_token_ids,       # list[int]  (may include eos id 1 and/or 106, …)
        model_device,         # Fixed: was 'device' in your call but 'model_device' in signature
        logger):

    # — generation kwargs aligned with SFT —
    gen_kwargs = {
        **inputs,                      # input_ids & attention_mask
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "pad_token_id": tokenizer.pad_token_id,   # Use pad_token_id like SFT script
        "eos_token_id": list(stop_token_ids),     # Pass the full list like your original
        "return_dict_in_generate": True,
        "output_hidden_states": True,
        "no_repeat_ngram_size": 3,
    }

    with torch.no_grad():
        gen_out = model.generate(**gen_kwargs)     # HF GenerateOutput

    return gen_out          # caller uses gen_out.sequences[0]

# --- Hook Function (Captures Module OUTPUT) ---
def capture_activation_hook(
    module,
    input_tensors,
    output_tensor,
    layer_index,         # HDF5 layer index (e.g., 0 for embeds, or l for transformer layer l)
    activation_type,     # e.g., "residual_stream", "attn_output", "mlp_output"
    storage_dict
):
    """
    PyTorch forward hook to capture and store activations.
    This function is designed to be partially configured with metadata (layer_index, etc.)
    and then registered on a specific module.
    """
    try:
        # The output of many transformer modules is a tuple; we're interested in the primary hidden state tensor.
        output_to_capture = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor

        if not isinstance(output_to_capture, torch.Tensor):
            logger.warning(f"Hook for L{layer_index}_{activation_type} received non-tensor output: {type(output_to_capture)}. Skipping.")
            return

        # Ensure we are working with batch size of 1, as generation is sequential.
        if output_to_capture.dim() == 3 and output_to_capture.shape[0] == 1:
            # Shape: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            activation_data = output_to_capture[0].detach().cpu().to(torch.float32).numpy()
        elif output_to_capture.dim() == 2:
            # Shape: (seq_len, hidden_dim) - handle cases where batch dim is already squeezed.
            activation_data = output_to_capture.detach().cpu().to(torch.float32).numpy()
        else:
            logger.error(f"Unexpected tensor shape for L{layer_index}_{activation_type}: {output_to_capture.shape}. Skipping.")
            return

        # Store the captured activation in the global dictionary.
        # The structure is {layer_index: {activation_type: [list_of_activations_per_step]}}
        if layer_index not in storage_dict:
            storage_dict[layer_index] = {}
        if activation_type not in storage_dict[layer_index]:
            storage_dict[layer_index][activation_type] = []
        
        storage_dict[layer_index][activation_type].append(activation_data)

    except Exception as e:
        logger.error(f"Error in hook for L{layer_index}_{activation_type}: {e}", exc_info=True)


# --- Core Capture Function ---
def run_capture(config_path, output_prefix=None, responses_only=False, csv_output=False):
    global current_prompt_run_activations, hook_handles

    logger.info(f"--- CI03 Capture Script v2 ---")
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from configuration file: {config_path}")
        return

    # Validate essential configuration keys
    required_keys = ['model_name', 'prompts_file', 'output_file']
    if not all(key in config for key in required_keys):
        logger.error(f"Config file missing one or more required keys: {required_keys}")
        return

    model_name = config['model_name']
    prompts_file = config['prompts_file']
    output_file = os.path.join(output_prefix, config['output_file']) if output_prefix else config['output_file']
    
    # Generation parameters
    runs_per_prompt = config.get('runs_per_prompt', 1)
    max_new_tokens = config.get('max_new_tokens', 1024) # Sensible default

    if responses_only:
        logger.info("--- Mode: Responses-Only ---")
        # Determine output format
        csv_output = args.csv_output
        if csv_output:
            if not output_file.endswith('.csv'):
                output_file = os.path.splitext(output_file)[0] + '.csv'
        else:
            if not output_file.endswith('.jsonl'):
                output_file = os.path.splitext(output_file)[0] + '.jsonl'
        
        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output will be saved to: {output_file}")

    else:
        logger.info("--- Mode: Full Activation Capture ---")
        if not output_file.endswith('.hdf5'):
             output_file = os.path.splitext(output_file)[0] + '.hdf5'
        logger.info(f"Output will be saved to: {output_file}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Load prompts from the specified JSON file
    try:
        with open(prompts_file, 'r') as f:
            prompts_data = json.load(f)
        if not isinstance(prompts_data, list):
             raise ValueError("Prompts file should contain a JSON list of prompt objects.")
        logger.info(f"Loaded {len(prompts_data)} prompts from {prompts_file}")
    except Exception as e:
        logger.error(f"Error reading or parsing prompts file {prompts_file}: {e}")
        return

    # Setup device and model data type
    if torch.cuda.is_available():
        device = torch.device("cuda")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        compute_dtype = torch.float32
    logger.info(f"Target device type: {device.type}")
    logger.info(f"Using compute dtype: {compute_dtype}")

    # Load Model and Tokenizer
    try:
        logger.info("Loading tokenizer...")
        
        # Handle local vs remote model loading
        architecture_type = config.get('architecture_type', 'remote')
        
        if architecture_type == 'local':
            # For local models, use model_path and load tokenizer from model_name
            model_path = config.get('model_path')
            if not model_path:
                logger.error("Local architecture specified but no 'model_path' provided in config")
                return
            
            if os.path.exists(model_path):
                logger.info(f"Loading local model from: {model_path}")
                model_name_for_loading = model_path
                tokenizer = AutoTokenizer.from_pretrained(model_name)  # Use model_name for tokenizer
            else:
                logger.error(f"Local model path not found: {model_path}")
                return
        else:
            # Default remote loading
            model_name_for_loading = model_name
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token; setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_for_loading,
            torch_dtype=compute_dtype,
            device_map="auto" # Automatically handle device placement
        )
        model.eval() # Set model to evaluation mode
        model_device = next(model.parameters()).device
        logger.info(f"Model loaded successfully ({type(model).__name__}). Primary device: {model_device}")

        # --- Set up robust stopping criteria ---
        # Instruction-tuned models have specific tokens to signal the end of their turn.
        # We create a list of these to pass to the generate function.
        stop_token_ids = set()
        if tokenizer.eos_token_id is not None:
            stop_token_ids.add(tokenizer.eos_token_id)

        # Add other known terminator tokens if they exist in the tokenizer's vocabulary.
        # This makes the script more robust across different model families (Gemma, Llama, etc.).
        known_terminators = ["<|eot_id|>", "<|end_of_text|>", "<end_of_turn>"]
        for token_str in known_terminators:
            # Use convert_tokens_to_ids to safely get the ID
            ids = tokenizer.convert_tokens_to_ids(token_str)
            # It might return a single ID or a list if the token is split.
            # We only care about single-token terminators here.
            if isinstance(ids, int) and ids != tokenizer.unk_token_id:
                stop_token_ids.add(ids)
        
        stop_token_ids = list(stop_token_ids)
        logger.info(f"Determined stop token IDs for generation: {stop_token_ids}")

    except Exception as e:
        logger.error(f"Failed to load model or tokenizer: {e}", exc_info=True)
        return

    # --- Main Processing Loop ---
    # Open the output file based on the selected mode
    output_file_handle = None
    if responses_only:
        # Overwrite file for simplicity in this mode, as it's for exploration
        output_file_handle = open(output_file, 'w')
    else:
        try:
            output_file_handle = h5py.File(output_file, 'a')
        except Exception as e:
            logger.error(f"Failed to open HDF5 file {output_file}: {e}")
            return

    total_prompts = len(prompts_data)
    for i, prompt_info in enumerate(prompts_data):
        prompt_text = prompt_info.get('text')
        prompt_id = prompt_info.get('prompt_id', str(uuid.uuid4()))

        if not prompt_text:
             logger.warning(f"Skipping prompt {i+1}/{total_prompts} (ID: {prompt_id}) due to missing 'text' field.")
             continue

        for run_idx in range(runs_per_prompt):
            logger.info(f"Processing prompt {i+1}/{total_prompts} (ID: {prompt_id}), Run: {run_idx + 1}/{runs_per_prompt}...")
            start_prompt_time = time.time()
            
            # Skip if data already exists (only checkable for HDF5 mode)
            if not responses_only:
                prompt_run_group_name = f"{prompt_id}/run_{run_idx}"
                if prompt_run_group_name in output_file_handle:
                    logger.warning(f"Data for Prompt ID {prompt_id}, Run {run_idx} already exists in HDF5. Skipping.")
                    continue

            # Clear state for this run
            current_prompt_run_activations.clear()
            hook_handles.clear()
            
            prompt_input_length_for_logging = -1

            try:
                # --- Hook Registration (only in full capture mode) ---
                if not responses_only:
                    logger.debug(f"Registering hooks for activation capture...")
                    core_model = getattr(model, "model", model)
                    layers_module_list = None
                    for attr in ['layers', 'block', 'blocks', 'h']:
                        if hasattr(core_model, attr):
                            layers_module_list = getattr(core_model, attr)
                            break
                    if layers_module_list is None:
                        raise AttributeError(f"Could not find transformer layers list in {type(core_model).__name__}")
                    
                    embed_tokens_module = getattr(core_model, 'embed_tokens', None)
                    if embed_tokens_module is None:
                         raise AttributeError(f"Could not find embed_tokens module in {type(core_model).__name__}")

                    num_layers = len(layers_module_list)
                    logger.debug(f"Found {num_layers} layers in {type(core_model).__name__}.")

                    # 1. Hook for x^(0) (initial embedding output)
                    hook_handles.append(embed_tokens_module.register_forward_hook(
                        partial(capture_activation_hook, layer_index=0, activation_type="residual_stream", storage_dict=current_prompt_run_activations)
                    ))

                    # 2. Hooks for attn and mlp outputs (the deltas for reconstruction)
                    for l_idx, layer in enumerate(layers_module_list):
                        attn_module = getattr(layer, 'self_attn', None)
                        if attn_module and hasattr(attn_module, 'o_proj'):
                            hook_handles.append(attn_module.o_proj.register_forward_hook(
                                partial(capture_activation_hook, layer_index=l_idx, activation_type="attn_output", storage_dict=current_prompt_run_activations)
                            ))
                        
                        mlp_module = getattr(layer, 'mlp', None)
                        if mlp_module and hasattr(mlp_module, 'down_proj'):
                             hook_handles.append(mlp_module.down_proj.register_forward_hook(
                                partial(capture_activation_hook, layer_index=l_idx, activation_type="mlp_output", storage_dict=current_prompt_run_activations)
                            ))
                    
                    logger.debug(f"Registered {len(hook_handles)} hooks in total.")

                # --- Tokenization and Generation ---
                messages = [{"role": "user", "content": prompt_text}]
                formatted_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)

                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model_device)
                prompt_input_length_for_logging = inputs["input_ids"].shape[1]  # Add this back!

                logger.debug(f"Generating response (max_new_tokens={max_new_tokens})...")
                forward_start_time = time.time()
                outputs = generate_response(
                        model, tokenizer,
                        inputs,                 # ← tokenised inputs
                        max_new_tokens, stop_token_ids,
                        model_device, logger)   # Fixed: was 'device', now 'model_device'
                logger.debug(f"Model generation took {time.time() - forward_start_time:.2f}s")

                # --- Process and Log Response ---
                generated_sequence_ids = outputs.sequences[0]
                newly_generated_ids = generated_sequence_ids[prompt_input_length_for_logging:]  # Now this works!
                generated_text = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)

                token_count = len(newly_generated_ids)
                logger.info(f"Generated {token_count} tokens (stopped naturally).")
                if token_count > 0:
                    logger.info(f"Response preview: '{generated_text.strip()[:150]}...'")
                else:
                    logger.warning("Empty response generated!")

            except Exception as e:
                logger.error(f"Error during model generation or hooking for prompt {prompt_id}, run {run_idx}: {e}", exc_info=True)
                continue # Skip to next run/prompt
            finally:
                # Always remove hooks after a run to prevent memory leaks
                if hook_handles: 
                    for handle in hook_handles:
                        handle.remove()
                    hook_handles.clear()

            # --- Save Results ---
            try:
                if responses_only:
                    if csv_output:
                        response_record = {
                            "prompt_id": prompt_id,
                            "run_id": f"run_{run_idx}",  # Match extract script format
                            "prompt_text": prompt_text,
                            "response_text": generated_text.strip()
                        }
                        csv_headers = ["prompt_id", "run_id", "response_text"]
                        append_to_csv(output_file, [response_record], csv_headers)

                    else:
                        response_record = {
                            "prompt_id": prompt_id,
                            "run_index": run_idx,
                            "model_name": model_name,
                            "prompt_text": prompt_text,
                            "generated_text": generated_text.strip(),
                            "token_count": token_count
                        }
                        output_file_handle.write(json.dumps(response_record) + '\n')
                
                else: # Full activation capture mode
                    if not current_prompt_run_activations:
                        logger.warning(f"No activations were captured. Skipping HDF5 save.")
                        continue
                    
                    logger.debug(f"Saving activations to HDF5 group: {prompt_run_group_name}...")
                    pr_group = output_file_handle.create_group(prompt_run_group_name)

                    # Save metadata as HDF5 attributes
                    for key, value in prompt_info.items():
                        try:
                            if value is None: value = 'None'
                            pr_group.attrs[key] = str(value) if isinstance(value, (list, dict)) else value
                        except TypeError:
                            pr_group.attrs[key] = str(value)
                    
                    pr_group.attrs['prompt_text'] = prompt_text
                    pr_group.attrs['prompt_input_length'] = prompt_input_length_for_logging
                    pr_group.attrs['generated_text'] = generated_text.strip()
                    pr_group.attrs['full_sequence_length'] = generated_sequence_ids.shape[0]
                    pr_group.attrs['token_count'] = token_count 

                    pr_group.create_dataset('token_ids', data=generated_sequence_ids.cpu().numpy(), compression="gzip")
                    
                    for layer_idx, layer_data in sorted(current_prompt_run_activations.items()):
                        layer_group = pr_group.create_group(f'layer_{layer_idx}')
                        for act_type, steps_data in layer_data.items():
                            act_type_group = layer_group.create_group(act_type)
                            for step_idx, step_activation_data in enumerate(steps_data):
                                act_type_group.create_dataset(
                                    f'step_{step_idx}', 
                                    data=step_activation_data,
                                    compression="gzip",
                                    dtype='float32'
                                )
                    output_file_handle.flush()
                    logger.info(f"Successfully saved activations to HDF5.")

            except Exception as e:
                logger.error(f"Error saving results for {prompt_id}, run {run_idx}: {e}", exc_info=True)
                if not responses_only and prompt_run_group_name in output_file_handle:
                     try:
                         del output_file_handle[prompt_run_group_name]
                     except Exception as del_e:
                         logger.error(f"Failed to remove incomplete HDF5 group {prompt_run_group_name}: {del_e}")

            end_prompt_time = time.time()
            logger.info(f"Finished prompt {prompt_id}, run {run_idx} in {end_prompt_time - start_prompt_time:.2f}s.")

    # --- Cleanup ---
    logger.info("Capture process complete.")
    if output_file_handle:
        output_file_handle.close()
        logger.info(f"Closed output file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CI03: Capture LLM activations or responses for geometric analysis. v2 with robust stopping.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the JSON configuration file."
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="data/",
        help="Optional directory prefix for the output file specified in the config."
    )
    parser.add_argument(
        "--responses-only",
        action="store_true",
        help="If set, only generate responses and save them to a .jsonl file. \nSkips activation capture entirely."
    )
    parser.add_argument(
        "--csv-output",
        action="store_true",
        default=None,
        help="If set and --responses-only set then generate .csv file."
    )
    args = parser.parse_args()
    
    run_capture(args.config_file, output_prefix=args.output_prefix, responses_only=args.responses_only, csv_output=args.csv_output)

