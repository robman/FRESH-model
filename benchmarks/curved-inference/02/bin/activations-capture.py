import argparse
import json
import logging
import os
import time
import uuid
from functools import partial

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# --- Global Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

current_prompt_run_activations = {} # Stores activations for the current prompt and run
hook_handles = [] # To store hook handles for removal

class AggressiveLineStoppingCriteria(StoppingCriteria):
    """
    Very aggressive stopping - stops at ANY sign of multi-line content
    """

    def __init__(self, tokenizer, min_response_length=6):
        self.tokenizer = tokenizer
        self.min_response_length = min_response_length
        self.initial_length = None

    def __call__(self, input_ids, scores, **kwargs):
        current_length = input_ids.shape[-1]

        if self.initial_length is None:
            self.initial_length = current_length
            return False

        new_tokens_count = current_length - self.initial_length

        # Must generate minimum response
        if new_tokens_count < self.min_response_length:
            return False

        try:
            # Decode only the newly generated tokens
            new_tokens = input_ids[0][self.initial_length:]
            new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Stop at ANY newline character immediately
            if any(char in new_text for char in ['\n', '\r']):
                return True
                
            # Stop if we see conversation patterns
            if any(pattern in new_text for pattern in ['Thought:', 'User:', 'Assistant:']):
                return True

        except Exception:
            pass

        # Earlier fallback
        if new_tokens_count >= 60:
            return True

        return False
            
# --- Generation Functions for Sleeper Agent Experiment ---
def generate_single_line_response(model, tokenizer, inputs, model_device, logger):
    """
    Generate single line Assistant responses
    """
    
    stopping_criteria = StoppingCriteriaList([
        AggressiveLineStoppingCriteria(tokenizer, min_response_length=8)
    ])
    
    generation_params = {
        **inputs,
        'max_new_tokens': 80,        # Reasonable limit for single line
        'min_new_tokens': 6,         
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 40,
        'repetition_penalty': 1.1,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'stopping_criteria': stopping_criteria,
        'return_dict_in_generate': True,
        'output_hidden_states': True,
    }
    
    with torch.no_grad():
        outputs = model.generate(**generation_params)
    
    return outputs

# --- Hook Function (captures module OUTPUT) ---
def capture_activation_hook(
    module,
    input_tensors,
    output_tensor,
    layer_index,         # This will be the HDF5 layer index (0 for embeds, l+1 for output of transformer layer l)
    activation_type,     # e.g., "residual_stream_layer_input", "attn_output", "mlp_output"
    storage_dict
):
    try:
        # Standard way to get the primary tensor output, handles modules returning tuples.
        output_to_capture = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor

        if not isinstance(output_to_capture, torch.Tensor):
            logger.warning(f"Hook for HDF5_L{layer_index}_{activation_type} received non-tensor output: {type(output_to_capture)}. Skipping.")
            return

        # Process based on tensor dimensions (expecting batch_size=1)
        if output_to_capture.dim() == 3 and output_to_capture.shape[0] == 1:
            # Shape: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
            activation_data = output_to_capture[0].detach().cpu().to(torch.float32).numpy()
        elif output_to_capture.dim() == 2:
            # Shape: (seq_len, hidden_dim) - if batch dim was already squeezed
            activation_data = output_to_capture.detach().cpu().to(torch.float32).numpy()
            logger.info(f"Captured 2D tensor for HDF5_L{layer_index}_{activation_type}. Shape: {activation_data.shape}")
        else:
            logger.error(f"Unexpected tensor shape for HDF5_L{layer_index}_{activation_type}: {output_to_capture.shape}. Skipping capture.")
            return

        # Store the activation
        if layer_index not in storage_dict:
            storage_dict[layer_index] = {}
        if activation_type not in storage_dict[layer_index]:
            storage_dict[layer_index][activation_type] = []
        
        storage_dict[layer_index][activation_type].append(activation_data)
        # logger.debug(f"Captured HDF5_L{layer_index}_{activation_type} (step {len(storage_dict[layer_index][activation_type])-1}), Shape: {activation_data.shape}")

    except Exception as e:
        logger.error(f"Error in hook for HDF5_L{layer_index}_{activation_type}: {e}", exc_info=True)


# --- Core Capture Function ---
def run_capture(config_path, output_prefix=None):
    global current_prompt_run_activations, hook_handles

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

    required_keys = ['model_name', 'prompts_file', 'output_hdf5_file', 'capture_points']
    if not all(key in config for key in required_keys):
        logger.error(f"Config file missing one or more required keys: {required_keys}")
        return

    model_name = config['model_name']
    prompts_file = config['prompts_file']
    output_hdf5_file = output_prefix+"/"+config['output_hdf5_file']
    capture_points = config['capture_points'] # Dict: {"attn_output": true, "mlp_output": true, "residual_stream_layer_input": true}
    batch_size = config.get('batch_size', 1)
    runs_per_prompt = config.get('runs_per_prompt', 1)

    if batch_size != 1:
        logger.error("Currently, only batch_size=1 is supported for activation capture during generation.")
        return

    logger.info(f"Model: {model_name}")
    logger.info(f"Output HDF5: {output_hdf5_file}")
    logger.info(f"Runs per prompt: {runs_per_prompt}")
    logger.info(f"Activations to capture: {', '.join(k for k, v in capture_points.items() if v)}")

    output_dir = os.path.dirname(output_hdf5_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    try:
        with open(prompts_file, 'r') as f:
            prompts_data = json.load(f)
        if not isinstance(prompts_data, list):
             raise ValueError("Prompts file should contain a JSON list of prompt objects.")
        logger.info(f"Loaded {len(prompts_data)} prompts from {prompts_file}")
    except Exception as e:
        logger.error(f"Error reading or parsing prompts file {prompts_file}: {e}")
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        compute_dtype = torch.float32
    logger.info(f"Target device type: {device.type}")
    logger.info(f"Using compute dtype: {compute_dtype}")

    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token; setting to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=compute_dtype,
            device_map="auto"
        )
        model.eval()
        model_device = next(model.parameters()).device
        logger.info(f"Model loaded successfully ({type(model).__name__}). Primary device: {model_device}")

        core_model = None
        if hasattr(model, "model"): # Standard for most HF models like Llama, Gemma
            core_model = model.model
            logger.info(f"Accessed core model ({type(core_model).__name__}) via 'model.model'.")
        else:
            core_model = model
            logger.warning(f"Could not access core model via 'model.model'. Using the top-level model ({type(core_model).__name__}) to find layers and embed_tokens.")

        layers_module_list = None
        candidate_layer_attr_names = ['layers', 'block', 'blocks', 'h', 'decoder_layers', 'transformer_layers']
        found_layers_attr = None

        for attr_name in candidate_layer_attr_names:
            if hasattr(core_model, attr_name):
                potential_layers = getattr(core_model, attr_name)
                if isinstance(potential_layers, torch.nn.ModuleList):
                    layers_module_list = potential_layers
                    found_layers_attr = attr_name
                    logger.info(f"Found layers list attribute '{attr_name}' in {type(core_model).__name__}.")
                    break
        if layers_module_list is None:
            logger.error(f"Could not find a ModuleList attribute for layers in {type(core_model).__name__} using candidates: {candidate_layer_attr_names}.")
            raise AttributeError(f"Failed to find the list of layers in {type(core_model).__name__}.")
        num_layers = len(layers_module_list) # This is L, the number of transformer blocks
        logger.info(f"Model has {num_layers} transformer layers (found in {type(core_model).__name__} via attribute '{found_layers_attr}').")

        embed_tokens_module = None
        if hasattr(core_model, 'embed_tokens'):
            embed_tokens_module = core_model.embed_tokens
            logger.info(f"Found embed_tokens module at core_model.embed_tokens ({type(embed_tokens_module).__name__})")
        else:
            logger.warning("Could not automatically find 'embed_tokens' module in core_model. x^(0) capture might fail.")


    except Exception as e:
        logger.error(f"Failed to load model, tokenizer, or find critical modules: {e}", exc_info=True)
        return

    logger.info(f"Opening HDF5 file for appending: {output_hdf5_file}")
    try:
        hdf5_file = h5py.File(output_hdf5_file, 'a')
    except Exception as e:
        logger.error(f"Failed to open HDF5 file {output_hdf5_file}: {e}")
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

            prompt_run_group_name = f"{prompt_id}/run_{run_idx}"
            if prompt_run_group_name in hdf5_file:
                logger.warning(f"Data for Prompt ID {prompt_id}, Run {run_idx} already exists in HDF5. Skipping.")
                continue

            current_prompt_run_activations.clear()
            hook_handles.clear()
            
            prompt_input_length_for_logging = -1

            try:
                logger.debug(f"Registering hooks for prompt {prompt_id}, run {run_idx}...")
                
                # Hook for x^(0) (embedding output)
                if capture_points.get("residual_stream_layer_input") and embed_tokens_module:
                    hook_func_embed = partial(
                        capture_activation_hook,
                        layer_index=0, # Store as HDF5 layer 0
                        activation_type="residual_stream_layer_input",
                        storage_dict=current_prompt_run_activations
                    )
                    handle = embed_tokens_module.register_forward_hook(hook_func_embed)
                    hook_handles.append(handle)
                    logger.debug(f"Registered hook for embed_tokens output (as HDF5 layer_0/residual_stream_layer_input).")

                # Hooks for transformer layers
                for l_idx in range(num_layers): # l_idx is 0 to L-1 (actual transformer layer index)
                    actual_layer_module = layers_module_list[l_idx]

                    # Hook for attn_output (delta_attn from layer l_idx)
                    if capture_points.get("attn_output"):
                        # Standard Llama/Gemma path: layer.self_attn.o_proj
                        # Ensure self_attn and o_proj exist
                        if hasattr(actual_layer_module, 'self_attn') and hasattr(actual_layer_module.self_attn, 'o_proj'):
                            module_to_hook = actual_layer_module.self_attn.o_proj
                            hook_func_attn = partial(
                                capture_activation_hook,
                                layer_index=l_idx, # Store with actual transformer layer index
                                activation_type="attn_output",
                                storage_dict=current_prompt_run_activations
                            )
                            handle = module_to_hook.register_forward_hook(hook_func_attn)
                            hook_handles.append(handle)
                        else:
                            logger.warning(f"Could not find self_attn.o_proj for layer {l_idx} to hook attn_output.")

                    # Hook for mlp_output (delta_mlp from layer l_idx)
                    if capture_points.get("mlp_output"):
                        # Standard Llama/Gemma path: layer.mlp.down_proj
                        if hasattr(actual_layer_module, 'mlp') and hasattr(actual_layer_module.mlp, 'down_proj'):
                            module_to_hook = actual_layer_module.mlp.down_proj
                            hook_func_mlp = partial(
                                capture_activation_hook,
                                layer_index=l_idx, # Store with actual transformer layer index
                                activation_type="mlp_output",
                                storage_dict=current_prompt_run_activations
                            )
                            handle = module_to_hook.register_forward_hook(hook_func_mlp)
                            hook_handles.append(handle)
                        else:
                            logger.warning(f"Could not find mlp.down_proj for layer {l_idx} to hook mlp_output.")

                    # Hook for x^(l_idx+1) (output of transformer layer l_idx)
                    if capture_points.get("residual_stream_layer_input"):
                        hook_func_residual_output = partial(
                            capture_activation_hook,
                            layer_index=l_idx + 1, # Output of layer l_idx is x^(l_idx+1), store as HDF5 layer_{l_idx+1}
                            activation_type="residual_stream_layer_input",
                            storage_dict=current_prompt_run_activations
                        )
                        handle = actual_layer_module.register_forward_hook(hook_func_residual_output)
                        hook_handles.append(handle)
                
                logger.debug(f"Registered {len(hook_handles)} hooks in total.")

                logger.debug("Tokenizing prompt...")
                inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings).to(model_device)
                input_ids_prompt_only = inputs["input_ids"]
                prompt_input_length_for_logging = input_ids_prompt_only.shape[1]

                logger.debug("Running sleeper agent generation...")
                #robman
                logger.debug("Running single line response generation...")
                forward_start_time = time.time()
                
                outputs = generate_single_line_response(model, tokenizer, inputs, model_device, logger)
                
                logger.debug(f"Model generation took {time.time() - forward_start_time:.2f}s")

                generated_sequence_ids = outputs.sequences[0]
                generated_text = tokenizer.decode(generated_sequence_ids[prompt_input_length_for_logging:], skip_special_tokens=True)
                
                # Simple quality analysis
                response_words = len(generated_text.split())
                response_chars = len(generated_text.strip())
                has_yes_no = any(word in generated_text.lower() for word in ['yes', 'no'])
                
                # Check if it's a clean single line
                line_count = len(generated_text.strip().split('\n'))
                is_single_line = line_count == 1
                
                logger.info(f"Generated {response_words} words ({response_chars} chars), "
                          f"Yes/No: {has_yes_no}, Single line: {is_single_line}")
                
                if response_chars > 0:
                    logger.info(f"Response: '{generated_text.strip()}'")
                else:
                    logger.error("Empty response generated!")
                #robman

            except Exception as e:
                logger.error(f"Error during model generation or hook processing for prompt {prompt_id}, run {run_idx}: {e}", exc_info=True)
                for handle in hook_handles: 
                    handle.remove()
                hook_handles.clear()
                continue
            finally:
                if hook_handles: 
                    logger.debug("Removing hooks in finally block...")
                    for handle in hook_handles:
                        handle.remove()
                    logger.debug(f"Removed {len(hook_handles)} hooks.")
                    hook_handles.clear()

            if not current_prompt_run_activations:
                 logger.warning(f"No activations captured for prompt {prompt_id}, run {run_idx}. Skipping HDF5 save.")
                 continue

            logger.debug(f"Saving activations for prompt {prompt_id}, run {run_idx} to HDF5...")
            try:
                pr_group = hdf5_file.create_group(prompt_run_group_name)

                for key, value in prompt_info.items():
                    try:
                        if value is None: value = 'None' 
                        elif isinstance(value, (list, tuple, dict)): value = str(value) 
                        elif not isinstance(value, (int, float, str, bytes, np.generic)): value = str(value)
                        pr_group.attrs[key] = value
                    except TypeError as attr_err:
                        logger.warning(f"Could not save attribute '{key}' (value: {value}, type: {type(value)}) for {prompt_run_group_name}. Error: {attr_err}. Saving as string.")
                        pr_group.attrs[key] = str(value) 

                pr_group.attrs['original_prompt_text'] = prompt_text
                pr_group.attrs['generated_text'] = generated_text
                pr_group.attrs['prompt_input_length'] = prompt_input_length_for_logging
                pr_group.attrs['full_sequence_length'] = generated_sequence_ids.shape[0]

                pr_group.create_dataset('token_ids', data=generated_sequence_ids.cpu().numpy(), compression="gzip")

                # Save captured activations (now includes hooked residuals)
                # Structure: current_prompt_run_activations[hdf5_layer_idx][act_type] = list_of_tensors_per_step
                for hdf5_layer_idx_save, layer_activations in sorted(current_prompt_run_activations.items()):
                    # hdf5_layer_idx_save is 0 for x^(0), 1 for x^(1) ... L for x^(L)
                    # or for deltas, it's actual_transformer_layer_idx (0 to L-1)
                    layer_group = pr_group.create_group(f'layer_{hdf5_layer_idx_save}')
                    for act_type, steps_data_list in layer_activations.items():
                        act_type_group = layer_group.create_group(act_type)
                        if not steps_data_list:
                            logger.warning(f"No data found for HDF5_L{hdf5_layer_idx_save}_{act_type} for prompt {prompt_id}, run {run_idx}. Skipping dataset creation.")
                            continue
                        for step_idx, step_activation_data in enumerate(steps_data_list):
                            act_type_group.create_dataset(
                                f'step_{step_idx}', 
                                data=step_activation_data,
                                compression="gzip",
                                dtype='float32'
                            )
                hdf5_file.flush()
                logger.info(f"Successfully saved activations for prompt {prompt_id}, run {run_idx}.")

            except Exception as e:
                logger.error(f"Error saving activations for {prompt_run_group_name} to HDF5: {e}", exc_info=True)
                if prompt_run_group_name in hdf5_file:
                     try:
                         del hdf5_file[prompt_run_group_name]
                         logger.info(f"Removed potentially incomplete group {prompt_run_group_name} from HDF5 due to saving error.")
                     except Exception as del_e:
                         logger.error(f"Failed to remove incomplete group {prompt_run_group_name} from HDF5: {del_e}")

            end_prompt_time = time.time()
            logger.info(f"Finished processing prompt {prompt_id}, run {run_idx} in {end_prompt_time - start_prompt_time:.2f} seconds.")

    logger.info("Closing HDF5 file.")
    hdf5_file.close()
    logger.info("Activation capture process complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture LLM activations during generation using PyTorch hooks for sleeper agent geometry experiment.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the JSON configuration file."
    )
    parser.add_argument("--output_prefix", type=str, default="data/", help="data dir prefix")
    args = parser.parse_args()
    run_capture(args.config_file, output_prefix=args.output_prefix)
