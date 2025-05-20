import argparse
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from utils import open_config, create_model
from detector.attn import AttentionDetector
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def set_seed(seed):
    """
    Sets the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_binary_label(data_item, dataset_name_str):
    """
    Attempts to extract a binary label (0 or 1) from a data item,
    checking known label columns for specific datasets.
    Returns 0, 1, or None if no suitable binary label is found.
    """
    label_value = None
    # Try common 'label' column first
    if 'label' in data_item:
        label_value = data_item['label']
        if isinstance(label_value, (int, float)) and label_value in [0, 1]:
            return int(label_value)
        if isinstance(label_value, bool):  # Handle boolean labels
            return 1 if label_value else 0


    # Dataset-specific label column checks
    if dataset_name_str.startswith("allenai/WildJailbreak"):
        if 'is_jailbreak' in data_item:
            is_jailbreak_val = data_item['is_jailbreak']
            if isinstance(is_jailbreak_val, bool):
                return 1 if is_jailbreak_val else 0
            if isinstance(is_jailbreak_val, (int, float)) and is_jailbreak_val in [0, 1]:
                return int(is_jailbreak_val)
        elif 'red_team_attempt_result' in data_item and isinstance(data_item['red_team_attempt_result'], str):
            if data_item['red_team_attempt_result'].upper() == 'SUCCESS':
                return 1
            elif data_item['red_team_attempt_result'].upper() == 'FAILURE':
                return 0

                # If after all checks, label_value is still not a binary 0 or 1, return None
    if label_value is not None and not (isinstance(label_value, (int, float)) and label_value in [0, 1]):
        # print(f"Warning: Label '{label_value}' from dataset '{dataset_name_str}' is not binary 0/1. Treating as missing for metrics.")
        return None

    return None  # Default if no known label column or convertible label is found


def main(args):
    """
    Main function to run the dataset evaluation.
    """
    set_seed(args.seed)

    dataset_identifier = args.dataset_name
    dataset_config_name = None
    if ',' in args.dataset_name:
        parts = args.dataset_name.split(',', 1)
        dataset_identifier = parts[0]
        dataset_config_name = parts[1]
        print(f"Using dataset identifier: {dataset_identifier} with config: {dataset_config_name}")
    else:
        print(f"Using dataset identifier: {dataset_identifier} (no specific config name provided in argument)")
        if dataset_identifier == "allenai/WildJailbreak":
            dataset_config_name = "train"
            print(f"Info: Defaulting to config '{dataset_config_name}' for dataset {dataset_identifier}")

    safe_dataset_name_for_path = dataset_identifier.replace("/", "_")
    if dataset_config_name:
        safe_dataset_name_for_path += f"_{dataset_config_name}"

    output_logs_dir = f"./result/{safe_dataset_name_for_path}"
    output_result_file = f"./result/{safe_dataset_name_for_path}/result.jsonl"
    output_log_file = f"{output_logs_dir}/{args.model_name}-{args.seed}.json"

    model_config_path = f"./configs/model_configs/{args.model_name}_config.json"

    print(f"Loading model config from: {model_config_path}")
    model_config = open_config(config_path=model_config_path)

    print(f"Creating model: {args.model_name}")
    model = create_model(config=model_config)
    model.print_model_info()

    print(
        f"Attempting to load dataset: '{dataset_identifier}', config/name: '{dataset_config_name}', trust_remote_code=True")
    try:
        dataset = load_dataset(dataset_identifier, name=dataset_config_name, trust_remote_code=True)
        print(f"Successfully called load_dataset. Type of loaded 'dataset' object: {type(dataset)}")
        if isinstance(dataset, dict):
            print(f"Loaded 'dataset' is a dictionary with keys: {list(dataset.keys())}")
        else:
            print(f"Loaded 'dataset' is not a dictionary. It's a {type(dataset)} object.")

        if dataset_identifier.startswith("allenai/WildJailbreak") and dataset_config_name:
            test_data = dataset
            print(
                f"Info: For {dataset_identifier} with config {dataset_config_name}, using the loaded dataset directly as test_data.")
        elif 'test' in dataset:
            test_data = dataset['test']
        elif 'validation' in dataset:
            print("Warning: 'test' split not found. Using 'validation' split instead.")
            test_data = dataset['validation']
        elif 'train' in dataset:
            print("Warning: 'test' or 'validation' split not found. Using 'train' split.")
            test_data = dataset['train']
        else:
            if not isinstance(dataset, dict) and hasattr(dataset, '__iter__'):
                print(
                    "Info: Dataset loaded is not a DatasetDict. Assuming it's iterable and using it directly as test_data.")
                test_data = dataset
            else:
                raise ValueError(
                    f"Dataset {dataset_identifier} (Config: {dataset_config_name}) does not have a 'test', 'validation', or 'train' split, nor is it a direct iterable dataset. Available keys if dict: {list(dataset.keys())}")

    except Exception as e:
        print(
            f"Error during dataset loading or processing for {dataset_identifier} (Config: {dataset_config_name}): {e}")
        print(
            "This can be due to issues with the dataset's remote loading script, resource limitations (disk/memory), or network problems.")
        print(
            "If this is 'allenai/WildJailbreak' or a similar dataset, ensure your environment has sufficient resources and try loading it in a minimal, isolated script to confirm.")
        return

    print(f"Initializing detector: AttentionDetector for model {args.model_name}")
    detector = AttentionDetector(model)
    print("===================")
    print(f"Using detector: {detector.name}")

    true_binary_labels = []
    predicted_detection_flags_int = []
    predicted_attack_probabilities = []

    logs = []
    skip_metrics_calculation = False
    missing_text_count = 0
    processed_item_count = 0

    print(f"Processing samples from dataset {dataset_identifier} (Config: {dataset_config_name or 'default'})...")
    # Check if test_data is iterable
    if not hasattr(test_data, '__iter__'):
        print(f"Error: test_data for {dataset_identifier} is not iterable. Type: {type(test_data)}. Cannot proceed.")
        return

    for data_item in tqdm(test_data, desc=f"Evaluating {args.model_name} on {args.dataset_name}"):
        processed_item_count += 1
        text_input_keys = ['text', 'prompt', 'query', 'instruction', 'goal', 'question', 'dialogue', 'conversation']
        text_input = None
        for key in text_input_keys:
            if key in data_item and isinstance(data_item[key], str):
                text_input = data_item[key]
                break

        if text_input is None:
            missing_text_count += 1
            continue

        # result from detector.detect() is a tuple: (detection_flag, details_dict)
        # details_dict might contain 'focus_score' and hopefully 'model_generated_text'
        detection_result_tuple = detector.detect(text_input)
        detect_flag = detection_result_tuple[0]
        details_dict = detection_result_tuple[1]


        raw_score = details_dict.get('focus_score', 0.5)
        attack_probability_score = 1 - raw_score

        binary_label = get_binary_label(data_item, args.dataset_name)

        if binary_label is None:
            pass
        else:
            true_binary_labels.append(binary_label)
            predicted_detection_flags_int.append(1 if detect_flag else 0)
            predicted_attack_probabilities.append(attack_probability_score)

        log_entry = {
            "input_text": text_input,
            "extracted_binary_label": binary_label if binary_label is not None else "N/A",
            "detected_as_attack": detect_flag,
            "attack_probability_score": attack_probability_score,
            "raw_detector_output": details_dict,  # Store the full raw output from detector
            "original_item_keys": list(data_item.keys())
        }
        logs.append(log_entry)

    if missing_text_count > 0:
        print(
            f"Warning: Skipped {missing_text_count} out of {processed_item_count} items due to missing or non-string text field.")

    if not logs and processed_item_count == 0:
        print("No data was processed (dataset might be empty or all items missed text field). Exiting.")
        return

    if not true_binary_labels:
        print(
            f"Warning: No valid binary labels found for dataset {args.dataset_name}. Metrics calculation will be skipped.")
        skip_metrics_calculation = True

    auc_score_rounded, auprc_score_rounded, fnr_rounded, fpr_rounded = "N/A", "N/A", "N/A", "N/A"
    tn, fp, fn, tp = "N/A", "N/A", "N/A", "N/A"

    if not skip_metrics_calculation and len(set(true_binary_labels)) < 2:
        print(
            f"Warning: Metrics calculation (AUC/AUPRC) skipped for {args.dataset_name} as all collected labels are the same: {set(true_binary_labels)}. Only confusion matrix might be informative.")
        if predicted_detection_flags_int:
            cm_labels = sorted(list(set(true_binary_labels + [0, 1])))
            if len(cm_labels) == 1: cm_labels = [0, 1]
            try:
                if len(true_binary_labels) == len(predicted_detection_flags_int):
                    tn_val, fp_val, fn_val, tp_val = confusion_matrix(true_binary_labels, predicted_detection_flags_int,
                                                                      labels=cm_labels).ravel()
                    tn, fp, fn, tp = int(tn_val), int(fp_val), int(fn_val), int(tp_val)
                    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0
                    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
                    fnr_rounded = round(fnr_val, 3)
                    fpr_rounded = round(fpr_val, 3)
                else:
                    print("Error: Mismatch in length of true labels and predictions for CM calculation.")
            except ValueError as cm_error:
                print(
                    f"Could not compute full confusion matrix components due to label distribution or other error: {cm_error}")
        skip_metrics_calculation = True

    if not skip_metrics_calculation:
        print("Calculating metrics...")
        if len(true_binary_labels) != len(predicted_attack_probabilities):
            print("Error: Mismatch in length of true labels and predicted probabilities. Skipping AUC/AUPRC.")
            skip_metrics_calculation = True
        elif len(true_binary_labels) == 0:
            print("Error: No true binary labels available for metric calculation. Skipping AUC/AUPRC.")
            skip_metrics_calculation = True
        else:
            auc_score = roc_auc_score(true_binary_labels, predicted_attack_probabilities)
            auprc_score = average_precision_score(true_binary_labels, predicted_attack_probabilities)

            if len(true_binary_labels) == len(predicted_detection_flags_int):
                cm_tn, cm_fp, cm_fn, cm_tp = confusion_matrix(true_binary_labels, predicted_detection_flags_int,
                                                              labels=[0, 1]).ravel()
                tn, fp, fn, tp = int(cm_tn), int(cm_fp), int(cm_fn), int(cm_tp)

                fnr = cm_fn / (cm_fn + cm_tp) if (cm_fn + cm_tp) > 0 else 0
                fpr = cm_fp / (cm_fp + cm_tn) if (cm_fp + cm_tn) > 0 else 0

                auc_score_rounded = round(auc_score, 3)
                auprc_score_rounded = round(auprc_score, 3)
                fnr_rounded = round(fnr, 3)
                fpr_rounded = round(fpr, 3)
            else:
                print("Error: Mismatch in length of true labels and predictions for final CM. Skipping FNR/FPR.")

    print(f"\nResults for {args.model_name} on {args.dataset_name} (Seed: {args.seed}):")
    print(f"AUC Score: {auc_score_rounded}")
    print(f"AUPRC Score: {auprc_score_rounded}")
    print(
        f"FNR: {fnr_rounded} (FN: {fn if isinstance(fn, int) else 'N/A'}, TP: {tp if isinstance(tp, int) else 'N/A'})")
    print(
        f"FPR: {fpr_rounded} (FP: {fp if isinstance(fp, int) else 'N/A'}, TN: {tn if isinstance(tn, int) else 'N/A'})")

    os.makedirs(output_logs_dir, exist_ok=True)

    print(f"Saving detailed logs to: {output_log_file}")
    metrics_dict = {
        "auc": auc_score_rounded, "auprc": auprc_score_rounded,
        "fnr": fnr_rounded, "fpr": fpr_rounded,
        "tn": tn if isinstance(tn, int) else "N/A",
        "fp": fp if isinstance(fp, int) else "N/A",
        "fn": fn if isinstance(fn, int) else "N/A",
        "tp": tp if isinstance(tp, int) else "N/A"
    }
    metrics_dict["results_count_processed"] = processed_item_count
    metrics_dict["results_count_logged"] = len(logs)
    metrics_dict["valid_labels_found_count"] = len(true_binary_labels)
    metrics_dict["missing_text_field_count"] = missing_text_count

    with open(output_log_file, "w") as f_out:
        # Save all logs, not just a preview
        json.dump({"config": vars(args),
                   "metrics": metrics_dict,
                   "results": logs},  # Changed "results_preview" to "results" and removed slicing
                  f_out, indent=4)

    print(f"Appending summary result to: {output_result_file}")
    summary_result = {
        "model": args.model_name,
        "dataset_spec": args.dataset_name,
        "seed": args.seed,
        "auc": auc_score_rounded,
        "auprc": auprc_score_rounded,
        "fnr": fnr_rounded,
        "fpr": fpr_rounded,
        "valid_labels_count": len(true_binary_labels),
        "processed_item_count": processed_item_count
    }
    with open(output_result_file, "a") as f_out:
        f_out.write(json.dumps(summary_result) + "\n")
    print("Evaluation complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Injection Detection Script using Attention Tracker")

    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model (should correspond to a config file in ./configs/model_configs/). E.g., 'qwen2-attn'")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset identifier, optionally with config (e.g., 'user/dataset,config' or 'allenai/WildJailbreak,train').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    main(args)
