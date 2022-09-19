# For discrete environment
history=1
context=10
kld=1.
adjd=10.0
datapath=traj_room_centre_len_16
datapath=traj_len_16

cp_path=context_${context}_history_${history}_kld_${kld} #_adjd_${adjd}

python main_boundary_vae.py \
  --num-epochs 1000 \
  --warmup_epochs 400 \
  --vae_state_size 2 \
  --vae_action_size 4 \
  --no-use_state_features \
  --expert-path ./h5_trajs/room_trajs/${datapath} \
  --checkpoint_every_epoch 50 \
  --results_dir ./results/vae/room_traj/discrete/${cp_path}/ \
  --log-interval 5 \
  --use_discrete_vae \
  --vae_context_size ${context} \
  --discrete_action \
  --run_mode train \
  --seed 0 \
  --env-type grid_room \
  --use_boundary \
  --cuda \
  --lambda_kld ${kld} | tee logs/${cp_path}.log 
  # --vae_history_size ${history} \
  # --use_history_in_policy 1
