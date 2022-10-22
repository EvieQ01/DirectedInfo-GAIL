#!/bin/bash
# for gen in 0.001 0.01  0.1 1. 3. 
# do
#     for post in 0.001 0.01 0.1 1. 3.
#     do
#         for dist in 0.001 0.01 0.1 1. 3.
#         do
#         python main_infoGAN.py --wandb  --lambda_post $post --lambda_dist $dist --lambda_gen $gen --teacher_force --cuda
#         done
#     done
# done
# for gen in  3 0.01 0.1 1. 0.001
# do
#     for post in 0.001 0.01 0.1 1. 3.
#     do
#         for dist in 0.001 0.01 0.1 1. 3.
#         do
#         python main_infoGAN.py --wandb --lambda_post $post --lambda_dist $dist --lambda_gen $gen --cuda
#         done
#     done
# done
# 这一部分可能重复了