export CUDA_VISIBLE_DEVICES=2
context=10
history=5
# path=centre_only_temp_1_0.1_context_4
path=context_${context}_history_${history}

args=(
  --vae_checkpoint_path ./results/vae/room_traj/discrete/${path}/checkpoint/cp_1000.pth
  --expert_path ./h5_trajs/room_trajs/traj_room_centre_len_50
  --results_dir ./results/room_traj/discrete/${path}
  --env-type 'grid_room'
  --discrete_action
  --context_size ${context}
  --num_epochs 2000
  --history_size ${history}
)
echo  "${args[@]}"
python -m pdb grid_world_gail.py  "${args[@]}"