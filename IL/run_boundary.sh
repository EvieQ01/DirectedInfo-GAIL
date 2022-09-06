# For discrete environment
history=1
context=10
kld=0.1
adjd=10.0
datapath=traj_room_centre_len_16
datapath=traj_len_16

cp_path=context_${context}_history_${history}_kld_${kld}_adjd_${adjd}

python main_boundary_vae.py \
  --use_rnn_goal 0 \
  --num-epochs 100 \
  --vae_state_size 2 \
  --vae_action_size 4 \
  --no-use_state_features \
  --expert-path ./h5_trajs/room_trajs/${datapath} \
  --checkpoint_every_epoch 50 \
  --results_dir ./results/vae/room_traj/discrete/${cp_path}/ \
  --log-interval 1 \
  --use_separate_goal_policy 0 \
  --use_goal_in_policy 0 \
  --use_discrete_vae \
  --vae_context_size ${context} \
  --vae_goal_size 10 \
  --discrete_action \
  --run_mode train \
  --seed 22 \
  --env-type grid_room \
  --use_boundary \
  --lambda_d_adjacent ${adjd} \
  --lambda_kld ${kld} | tee logs/${cp_path}.log
  # --vae_history_size ${history} \
  # --use_history_in_policy 1
