#!/bin/bash

# 参数设置
methods=("gmf" "mlp" "neumf")
negs=(0 1 2 3 4)

# 定义每个方法对应的CUDA设备
declare -A method_to_gpu
method_to_gpu=( ["gmf"]=0 ["mlp"]=1 ["neumf"]=2 )

# 遍历每一个组合
for neg in "${negs[@]}"; do
  for method in "${methods[@]}"; do

    cuda_device=${method_to_gpu[$method]}
    
    # 生成输出文件名
    output_file="${method}-${neg}.out"
    
    # 运行Python脚本
    CUDA_VISIBLE_DEVICES=$cuda_device nohup python train.py --method $method --factor 8 --neg $neg > $output_file &
    
    # 等待一小段时间以避免过多的并发请求
    sleep 1
  done
done
