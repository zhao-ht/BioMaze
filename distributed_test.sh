#!/bin/bash

## Check if the required arguments are provided
#if [ "$#" -ne 3 ]; then
#    echo "Usage: $0 <Meta-Llama-3-8B-Instruct> <alfworld_put> <codeact>"
#    exit 1
#fi

Model_name="$1"
Task_name="$2"
Method="$3"
Process_per_lm="$4"
Start_test_id="$5"
Temp="$6"
TEST_TO_FILE="$7"
Num_LM=$8


# Set CKPT_DIR and SAVE_DIR based on MODEL_DIR
if [[ $Model_name == "Meta-Llama-3.1-70B-Instruct" || $Model_name == "Qwen2.5-72B-Instruct" ]]; then
  Num_GPU_per_LM=8
else
  Num_GPU_per_LM=1
fi


# Calculate the values
Total_process=$(( Process_per_lm * $Num_LM ))
Max_index=$(( Total_process - 1 ))



##Launch vLLM Servers
#for i in $(seq 0 $((Num_LM-1))); do
#  # Calculate the start GPU ID for the current LM
#  start_gpu=$((i * Num_GPU_per_LM))
#
#  # Generate the list of GPU IDs for this LM
#  gpus=$(seq -s, $start_gpu $((start_gpu + Num_GPU_per_LM - 1)))
#
#  # Convert to string format (e.g., '0,1,2,3' or '4,5')
#  gpu_ids=$(echo $gpus | sed 's/ /,/g')
#
#  # Run the downstream test with the appropriate GPU IDs
#  echo $gpu_ids
#  ./downstream_test.sh "$i" "$gpu_ids" "$Model_name" "$TEST_TO_FILE"
#done

# Run downstream tasks
for i in $(seq 0 $Max_index); do
    port=$(( 8000 +i / Process_per_lm ))
    # when all the arguments are given, the test_id is only the agent tmux windows number
    test_id=$(( i+Start_test_id ))
    dis_id=$(( i ))
    echo $test_id
    ./downstream_test.sh "$test_id" "'0'" "$Model_name" "$TEST_TO_FILE" "$Task_name" "$Method" "$port" "$Temp" "$dis_id" "$Total_process"
done
