#!/bin/bash

# The list of models to evaluate
# MODELS=( "granite3_8b-attn" "llama3_8b-attn" "mistral_7b-attn" "gemma2_9b-attn" "phi3-attn" "qwen2-attn" )
MODELS=( "granite3_8b-attn" "phi3-attn" "qwen2-attn")

# Define the list of datasets to evaluate on.
# These should be valid Hugging Face dataset identifiers or paths to local datasets.
DATASETS=(
    "deepset/prompt-injections"
#    "allenai/WildJailbreak"
    # Add other datasets here if needed
)

SEED=42 # Define a seed, can also be iterated if needed

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "-----------------------------------------------------"
    echo "Evaluating Model: ${MODEL}"
    echo "-----------------------------------------------------"

    # Loop through each dataset for the current model
    for DATASET_NAME in "${DATASETS[@]}"; do
        echo "==> Dataset: ${DATASET_NAME}"

        # Run the Python script with the current model and dataset
        python run_dataset.py \
                        --model_name "${MODEL}" \
                        --dataset_name "${DATASET_NAME}" \
                        --seed ${SEED}

        echo "==> Finished ${MODEL} on ${DATASET_NAME}"
        echo ""
    done
    echo "-----------------------------------------------------"
    echo "Finished all datasets for Model: ${MODEL}"
    echo "-----------------------------------------------------"
    echo ""
done

echo "All evaluations complete."
