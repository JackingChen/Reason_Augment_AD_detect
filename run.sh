#!/bin/bash

# Default values
device=3
stage=0
# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --stage)
                stage="$2"
                shift 2
                ;;
            --device)
                device="$2"
                shift 2
                ;;
            --N)
                N="$2"
                shift 2
                ;;
            *)
                echo "未知的選項: $1"
                exit 1
                ;;
        esac
    done
}

# Enable debugging and exit on error
set -x
set -e

# Disable tokenizers parallelism
export TOKENIZERS_PARALLELISM=false

# Parse command line arguments
parse_args "$@"

# Rest of your script...

audioInfile_root=./saves/results
# infile=(data2vec-audio-large-960h_test.csv data2vec-audio-large-960h_train.csv)
infile=(data2vec-audio-large-960h_train.csv data2vec-audio-large-960h_dev.csv data2vec-audio-large-960h_test.csv)
# infile=(data2vec-audio-large-960h_train.csv)



export CUDA_VISIBLE_DEVICES=$device
# wv 跟 gr都會壞掉，不要跑
if [ "$stage" -le 1 ]; then
    python 0207_DM_SentenceLvl2inputHeterogeneous.py --inp1_embed albert-base-v1 --inp2_embed psych_ver_1.1__max_emb --epochs 5 --hidden_size 512 --sampleImbalance
fi
