
# generate and evaluate HumanEval in this script

# set your model path that can be read by .from_pretrained() here 
model=<model name or path>

# temperature, if using greedy decode, temp=0.0
temp=0.0
max_len=2048

# Number of predictions, for fast greedy_decode setting, set to 1, for standard setting, set to 200
pred_num=1
# generation batch size, set it according to your GPU memory, but smaller than pred_num
num_seqs_per_iter=1

# temperary generation path
output_path=preds/MBPP_T${temp}_N${pred_num}
mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model


# 500 problems
index=0
gpu_num=4
question_per_gpu=$(echo "scale=0; ( 500 + $gpu_num - 1 ) / $gpu_num" | bc)  
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * question_per_gpu))
  end_index=$(((i + 1) * question_per_gpu))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python mbpp_gen.py --model ${model} \
      --start_index ${start_index} --end_index ${end_index} --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path} --vllm &
  ) 
  # if (($index % $gpu_num == 0)); then wait; fi
done


save_path=../generation_for_harness/MBPP_T${temp}_N${pred_num}
mkdir -p ../generation_for_harness

echo 'Output path: '$output_path
python process_mbpp.py --path ${output_path} --out_path ${save_path}.jsonl --add_prompt
