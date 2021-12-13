#!/bin/bash

for N in 5 10 20 ; do
    for K in 5; do
        echo "[INFO] ..start jobs for proto ($N-way $K-shot)!"
        python main.py --model proto --dataset miniimagenet \
        --num_workers 1 --N $N --K $K --Q 15 \
        --total_train_episode_num 60000 --train_iter 5000 \
        --val_iter 5000 --test_iter 600 
        echo "[INFO] All jobs are done for proto ($N-way $K-shot)...!"
    done
done
