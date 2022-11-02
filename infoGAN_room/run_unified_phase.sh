# bs=32
# export CUDA_VISIBLE_DEVICES=0
# python main_infoGAN_unified.py --wandb --lambda_gen 0.01 --lambda_post 3.0 --lambda_dist 0.001 --batch_size $bs --cuda
# export CUDA_VISIBLE_DEVICES=1
# python main_infoGAN_unified.py --wandb --lambda_gen 0.01 --lambda_post 3.0 --lambda_dist 0.01 --batch_size $bs --cuda
# export CUDA_VISIBLE_DEVICES=2
# python main_infoGAN_unified.py --wandb --lambda_gen 0.01 --lambda_post 3.0 --lambda_dist 0.1 --batch_size $bs --cuda
# export CUDA_VISIBLE_DEVICES=3
# python main_infoGAN_unified.py --wandb --lambda_gen 0.01 --lambda_post 3.0 --lambda_dist 1 --batch_size $bs  --cuda


# # debug
python main_infoGAN_unified.py  --lambda_gen 0.1 --lambda_post 10 --lambda_dist 1. --batch_size 64 --threshold 0.3 --cuda --wandb --seed 2

# sweep 
bs=64
th=0.3
for lambda_post in 1. 3. 10 30
do
    for lambda_gen in 0.01 0.1 1. 10
    do
        for lambda_dist in 0.01 0.1 1. 10
        do
            python main_infoGAN_unified.py --wandb --lambda_gen $lambda_gen --lambda_post $lambda_post --lambda_dist $lambda_dist --batch_size $bs --threshold $th --cuda --lr 0.01 --distance_type l2  #--teacher_force
        done
    done
done