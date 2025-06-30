import argparse
import json
import logging
import os
import sys
from pathlib import Path
import h5py
import pandas as pd

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CSV helper
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Core analysis routine
# -----------------------------------------------------------------------------
def extract_prompt_response_text(reduction_config_path: str,
                                output_base_name: str,
                                specific_prompt_ids: list[str] = None):

    logger.info(f"Loading config: {reduction_config_path}")
    try:
        with open(reduction_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse config {reduction_config_path}: {e}")
        return

    hdf5_path = config.get('input_hdf5_file')
    model_name = config.get('model_name', 'unknown_model')

    if not hdf5_path:
        logger.error("Config missing 'input_hdf5_file'.")
        return

    if not Path(hdf5_path).exists():
        logger.error(f"HDF5 file not found: {hdf5_path}")
        return

    output_csv_file = f"{output_base_name}-prompt-response-text.csv"
    Path(output_csv_file).parent.mkdir(parents=True, exist_ok=True)

    csv_headers = [
        #"config_basename", "model_name", 
        "prompt_id", "run_id", 
        #"prompt_text", "response_text", "full_text"
        "prompt_text",
        "response_text"
    ]

    config_basename = Path(reduction_config_path).stem
    results_buffer = []

    try:
        with h5py.File(hdf5_path, 'r') as f_hdf5:
            prompt_ids_to_process = []
            if specific_prompt_ids:
                for pid_spec in specific_prompt_ids:
                    if pid_spec in f_hdf5:
                        prompt_ids_to_process.append(pid_spec)
                    else:
                        logger.warning(f"Specified prompt_id '{pid_spec}' not found in HDF5 file. Skipping.")
            else:
                prompt_ids_to_process = list(f_hdf5.keys())
            
            if not prompt_ids_to_process:
                logger.warning("No prompt IDs to process. Exiting.")
                return
            
            logger.info(f"Processing {len(prompt_ids_to_process)} prompt(s) for text extraction.")

            for pid_idx, pid in enumerate(prompt_ids_to_process):
                if pid not in f_hdf5:
                    logger.warning(f"Prompt ID {pid} not found in HDF5. Skipping.")
                    continue
                
                logger.info(f"Starting prompt {pid_idx + 1}/{len(prompt_ids_to_process)}: {pid}")
                prompt_group = f_hdf5[pid]
                run_ids = list(prompt_group.keys())

                for run_idx, run_id in enumerate(run_ids):
                    logger.debug(f"  Processing run {run_idx + 1}/{len(run_ids)}: {run_id}")
                    run_group = prompt_group[run_id]

                    # Debug: List all attributes in this run_group
                    logger.debug(f"    Available attributes in {pid}/{run_id}: {list(run_group.attrs.keys())}")

                    # Extract text data
                    #prompt_text = None
                    response_text = None
                    #full_text = None

                    # Try to get prompt text (T_p) from attributes
                    try:
                        if 'prompt_text' in run_group.attrs:
                            prompt_text_raw = run_group.attrs['prompt_text']
                            if isinstance(prompt_text_raw, bytes):
                                prompt_text = prompt_text_raw.decode('utf-8')
                            else:
                                prompt_text = str(prompt_text_raw)
                            logger.debug(f"    Extracted prompt_text: {len(prompt_text)} characters")
                        else:
                            logger.warning(f"    No 'prompt_text' attribute found for {pid}/{run_id}")
                    except Exception as e:
                        logger.error(f"    Error extracting prompt_text for {pid}/{run_id}: {e}")

                    # Try to get response text (T_r) from attributes
                    try:
                        if 'generated_text' in run_group.attrs:
                            response_text_raw = run_group.attrs['generated_text']
                            if isinstance(response_text_raw, bytes):
                                response_text = response_text_raw.decode('utf-8')
                            else:
                                response_text = str(response_text_raw)
                            response_text = response_text.strip()
                            logger.debug(f"    Extracted response_text: {len(response_text)} characters")
                        else:
                            logger.warning(f"    No 'generated_text' attribute found for {pid}/{run_id}")
                    except Exception as e:
                        logger.error(f"    Error extracting generated_text for {pid}/{run_id}: {e}")

                    # Construct full text if we have both parts
                    try:
                        if prompt_text and response_text:
                            full_text = prompt_text + response_text
                            logger.debug(f"    Constructed full_text: {len(full_text)} characters")
                    except Exception as e:
                        logger.error(f"    Error constructing full_text for {pid}/{run_id}: {e}")
                    
                    # Clean up text to avoid CSV formatting issues
                    if prompt_text:
                        prompt_text = prompt_text.replace('\r\n', '\n').replace('\r', '\n')
                    if response_text:
                        response_text = response_text.replace('\r\n', '\n').replace('\r', '\n')
                    if full_text:
                        full_text = full_text.replace('\r\n', '\n').replace('\r', '\n')
                    
                    record = {
                        #"config_basename": config_basename,
                        #"model_name": model_name,
                        "prompt_id": pid,
                        "run_id": run_id,
                        "prompt_text": prompt_text,
                        "response_text": response_text,
                        #"full_text": full_text
                    }

                    results_buffer.append(record)

                    # Periodic buffer flush
                    if len(results_buffer) >= 100:
                        logger.info(f"    Writing {len(results_buffer)} records to CSV...")
                        append_to_csv(output_csv_file, results_buffer, csv_headers)
                        results_buffer.clear()

            # Write final buffer
            if results_buffer:
                logger.info(f"Writing final {len(results_buffer)} records to CSV...")
                append_to_csv(output_csv_file, results_buffer, csv_headers)
                results_buffer.clear()

    except FileNotFoundError:
        logger.error(f"Input HDF5 file not found: {hdf5_path}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        if results_buffer:
            logger.error(f"Attempting to save {len(results_buffer)} buffered results due to error...")
            append_to_csv(output_csv_file, results_buffer, csv_headers)
        return

    logger.info(f"Text extraction complete. Results saved to {output_csv_file}")

    # Summary statistics
    try:
        result_df = pd.read_csv(output_csv_file)
        logger.info(f"Summary: Extracted text for {len(result_df)} runs across {len(result_df['prompt_id'].unique())} prompts")
        
        # Count non-null values for each text type
        #prompt_count = result_df['prompt_text'].notna().sum()
        response_count = result_df['response_text'].notna().sum()
        #full_count = result_df['full_text'].notna().sum()
        
        #logger.info(f"  Prompt texts extracted: {prompt_count}/{len(result_df)}")
        logger.info(f"  Response texts extracted: {response_count}/{len(result_df)}")
        #logger.info(f"  Full texts extracted: {full_count}/{len(result_df)}")
        
    except Exception as e:
        logger.warning(f"Could not generate summary statistics: {e}")

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract prompt and response text from HDF5 activation files for answer classification.")
    parser.add_argument("reduction_config_file", type=str, help="Path to JSON config (contains HDF5 input path and model name).")
    parser.add_argument("--output_base", type=str, required=True, help="Prefix for output CSV (e.g., 'data/model_name/prompt_set_name').")
    parser.add_argument("--prompt_ids", type=str, default=None, help="Comma-separated subset of prompt ids to process. If None, all prompts are processed.")

    args = parser.parse_args()
    pid_list = [p.strip() for p in args.prompt_ids.split(',')] if args.prompt_ids else None

    extract_prompt_response_text(
        reduction_config_path=args.reduction_config_file,
        output_base_name=args.output_base,
        specific_prompt_ids=pid_list
    )
