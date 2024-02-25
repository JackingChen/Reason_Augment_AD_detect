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
            --selected_psych)
                selected_psych="$2"
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

# selected_psychs=(psych_ver_1.1 psych_ver_2.1 Positive_attributes)
# selected_psychs=(psych_ver_2.1 Positive_attributes)
# selected_psychs=(psych_ver_3) #ver3 please use another script
selected_psychs=(psych_ver_1.1_highcorr)

audioInfile_root=/home/FedASR/dacs/centralized/saves/results
# infile=(data2vec-audio-large-960h_test.csv data2vec-audio-large-960h_train.csv)
infile=(data2vec-audio-large-960h/train.csv data2vec-audio-large-960h/dev.csv data2vec-audio-large-960h/test.csv)
# infile=(data2vec-audio-large-960h_train.csv)

if [ "$stage" -le 0 ]; then
    for selected_psych in ${selected_psychs[@]};do
        for inp in ${infile[@]};do 
            python Extract_Session_text.py --input_file ${audioInfile_root}/$inp --selected_psych $selected_psych; 
        done
    done
fi


if [ "$stage" -le 1 ]; then
    for selected_psych in ${selected_psychs[@]};do
        python PsychSummary2Embedding.py --prompt_dir text_data2vec-audio-large-960h_Phych-${selected_psych}
    done
fi