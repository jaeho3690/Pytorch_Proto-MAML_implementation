#!/bin/bash

for N in 5 10 20; do
    for K in 1 5; do
        echo "[INFO] ..start jobs for maml ($N-way $K-shot)!"
        # Test on 5way-1shot setting
        python main.py --model maml --dataset miniimagenet \
        --num_workers 1 --N $N --K $K --Q 15 --valtest_K 1\
        --total_train_episode_num 60000 --train_iter 5000 \
        --val_iter 5000 --test_iter 600 & exp1=$!

        # Test on 5way-5shot setting
        python main.py --model maml --dataset miniimagenet \
        --num_workers 1 --N $N --K $K --Q 15 --valtest_K 5\
        --total_train_episode_num 60000 --train_iter 5000 \
        --val_iter 5000 --test_iter 600 & exp2=$!

        wait "$exp1" "$exp2"
        echo "[INFO] All jobs are done for maml ($N-way $K-shot)...!"
    done
done

