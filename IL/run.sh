# For discrete environment
history=1
context=10
datapath=traj_room_centre_len_16
datapath=traj_len_16
cp_path=context_${context}_history_${history}

python -m pdb train_vae.py \
  --use_rnn_goal 0 \
  --num-epochs 1000 \
  --vae_state_size 2 \
  --vae_action_size 4 \
  --no-use_state_features \
  --expert-path ./h5_trajs/room_trajs/${datapath} \
  --checkpoint_every_epoch 50 \
  --results_dir ./results/vae/room_traj/discrete/context_${context}_history_${history}/ \
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
  --checkpoint_path ./results/vae/room_traj/discrete/${cp_path}/checkpoint/cp_1000.pth
  # --vae_history_size ${history} \
  # --use_history_in_policy 1
