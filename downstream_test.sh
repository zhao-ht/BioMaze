#!/bin/bash

TEST_ID=$1
CUDA_VISIBLE_DEVICES=$2
MODEL_DIR=$3
TEST_TO_FILE=$4
TASK=$5
METHOD=$6
CUSTOM_PORT=$7
TEMP=$8
CUSTOM_DIS_ID=$9
CUSTOM_TOTAL_DIS=${10}

# Convert CUDA_VISIBLE_DEVICES to an array
IFS=',' read -r -a cuda_array <<< "$CUDA_VISIBLE_DEVICES"

# Get the length of the array
NUM_DEVICE=${#cuda_array[@]}

# Set CKPT_DIR and SAVE_DIR based on MODEL_DIR

# Path to the YAML file
YAML_FILE="backbone/model_dir_config.yaml"

# Extract the path using awk
CKPT_DIR=$(awk -v model="$MODEL_DIR" '$1 == model ":" {print $2}' "$YAML_FILE" | tr -d '"')

if [ -z "$CKPT_DIR" ]; then
    echo "Error: Model path not found for $MODEL_DIR"
    exit 1
fi

mkdir -p outlog

# Compute the Port
if [ -z "$CUSTOM_PORT" ]; then
    PORT=$((8000+TEST_ID))
    KILL_COMMAND=""
#    KILL_COMMAND="&& tmux send-keys -t server$TEST_ID C-c"

    echo "Loading ckpt from $CKPT_DIR"

    echo "Launching vLLM server..."
    # Remove the log file if it exists
    LOG_FILE="outlog/server_output_${TEST_TO_FILE}_$TEST_ID.txt"
    rm -f $LOG_FILE

    # Launch vllm server
    if [ "$MODEL_DIR" == "Mistral-7B-v0.1" ]; then
      DTYPE="float16"
    else
      DTYPE="auto"
    fi

    echo "Launching vLLM in TMUTEST_ID server$TEST_ID with Port $PORT on GPU $CUDA_VISIBLE_DEVICES, DTYPE: $DTYPE"

    echo "Server output written to $LOG_FILE"
    CMD="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES nohup python -m vllm.entrypoints.openai.api_server --port $PORT --model $CKPT_DIR --served-model-name $MODEL_DIR --tensor-parallel-size $NUM_DEVICE --dtype $DTYPE &> $LOG_FILE &"
    echo $CMD
    eval $CMD

    # Wait for server startup
    while ! grep -q "Application startup complete" $LOG_FILE; do
        sleep 1
    done

    echo "vLLM server has launched"

else
    PORT=$CUSTOM_PORT
    KILL_COMMAND=""
    echo "Using vLLM with Port $PORT"
fi

# Compute distributed id

# Compute the Port
if [ -z "$CUSTOM_DIS_ID" ]; then
    DIS_ID=$TEST_ID
else
    DIS_ID=$CUSTOM_DIS_ID
fi
echo "Distributed ID $DIS_ID"

# Compute the total distributed number
if [ -z "$CUSTOM_TOTAL_DIS" ]; then
    TOTAL_DIS=8
else
    TOTAL_DIS=$CUSTOM_TOTAL_DIS
fi
echo "Total Distributed ID $TOTAL_DIS"

# Compute the output file name

echo $TEST_TO_FILE
TEST_LOG_FILE="outlog/test_${TEST_TO_FILE}_$TEST_ID.txt"
TEST_OUTPUT_CMD=" &> $TEST_LOG_FILE"
CMD_START="nohup"
CMD_END=" &"
echo "Test output written to $TEST_LOG_FILE"

# Execute script in agentTEST_ID tmux window based on TASK
if  [[ $TASK == "biomaze_judge"* ]]; then
  if [ "$METHOD" = "cot" ]; then
      Script="python -u downstream_test.py --exp_id 0 --dataset_name $TASK --planning_method cot --model_name $MODEL_DIR --host $PORT --resume --in_context_num 2 --answer_type judge --enable_cot --temperature $TEMP --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND $TEST_OUTPUT_CMD"

  elif [ "$METHOD" = "graph_agent" ]; then
      Script="python -u downstream_test.py --exp_id 0 --dataset_name $TASK --planning_method graph_agent --model_name $MODEL_DIR --host $PORT --answer_method conclusion --remove_uncertainty  --uncertainty_query --cot_merge_method uncertain --answer_type judge --temperature $TEMP --resume --max_steps 20 --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND $TEST_OUTPUT_CMD"

  elif [ "$METHOD" = "tog" ]; then
      Script="python -u downstream_test.py --exp_id 0 --dataset_name $TASK  --planning_method tog --model_name $MODEL_DIR --host $PORT  --max_length 1024 --temperature_exploration $TEMP --temperature_reasoning 0 --width 3 --depth 6 --remove_unnecessary_rel True --num_retain_entity 5 --prune_tools llm --resume  --answer_type judge --answer_method conclusion  --remove_uncertainty --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND  $TEST_OUTPUT_CMD"

  elif [ "$METHOD" = "cok" ]; then
      Script="python -u downstream_test.py --exp_id 0 --dataset_name $TASK  --planning_method cok --model_name $MODEL_DIR --host $PORT --resume --max_pieces 3  --in_context_num 2  --max_length 1024 --answer_type judge  --temperature $TEMP --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND  $TEST_OUTPUT_CMD"
  fi

elif  [[ $TASK == "biomaze_reasoning"* ]]; then
  if [ "$METHOD" = "cot" ]; then
      Script="python -u downstream_test.py --exp_id 0 --dataset_name $TASK --planning_method cot --model_name $MODEL_DIR --host $PORT --resume --in_context_num 2 --answer_type reasoning --no_evaluation --enable_cot --temperature $TEMP --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND  $TEST_OUTPUT_CMD"

  elif [ "$METHOD" = "graph_agent" ]; then
      Script="python -u downstream_test.py --exp_id 0 --dataset_name $TASK --planning_method graph_agent --model_name $MODEL_DIR --host $PORT --answer_method conclusion --remove_uncertainty --uncertainty_query --cot_merge_method uncertain --answer_type reasoning  --no_evaluation --temperature $TEMP --resume --max_steps 20 --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND  $TEST_OUTPUT_CMD"

  elif [ "$METHOD" = "tog" ]; then
      Script="python -u downstream_test.py --exp_id 0 --dataset_name $TASK  --planning_method tog --model_name $MODEL_DIR --host $PORT  --max_length 1024 --temperature_exploration $TEMP --temperature_reasoning  0 --width 3 --depth 6 --remove_unnecessary_rel True --num_retain_entity 5 --prune_tools llm --resume  --answer_type reasoning --no_evaluation --answer_method conclusion  --remove_uncertainty --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND  $TEST_OUTPUT_CMD"

  elif [ "$METHOD" = "cok" ]; then
      Script="python -u downstream_test.py --exp_id 0 --dataset_name $TASK  --planning_method cok --model_name $MODEL_DIR --host $PORT --resume --max_pieces 3  --in_context_num 2  --max_length 1024 --answer_type reasoning --no_evaluation --temperature $TEMP --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND  $TEST_OUTPUT_CMD"
  fi

fi

eval "$CMD_START $Script $CMD_END"

