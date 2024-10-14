#!/bin/bash

export LOG_DIR=/gpt3/logs
export BPE_DIR=/gpt3/dataset/bpe
export CONT=cr.ai.nebius.cloud/crnbu823dealq64cp1s6/megatron:mlcommons00f04c52-pytorch23.10-2

echo "Clear previous logs"
./rmlogs.sh

echo "Submit training job"
sbatch run_gpt3.sh $LOG_DIR $BPE_DIR $CONT
