#export TRAIN_DIR='./tune_results/policy_lr0.000247_dr0.2945_noise0.000225'
# sed "s#\${TRAIN_DIR}#$TRAIN_DIR#g" policy_new.yml > policy_new_tmp.yml
#CUDA_VISIBLE_DEVICES=1 python train_new.py --config policy_new_tmp.yml
#CUDA_VISIBLE_DEVICES=1 python train_new.py --config policy_new_tmp.yml
# export TRAIN_DIR='./seed_results/policy_lr1e-5_trial_2'
# sed "s#\${TRAIN_DIR}#$TRAIN_DIR#g" policy_new.yml > policy_new_tmp.yml
# CUDA_VISIBLE_DEVICES=1 python train_new.py --config policy_new_tmp.yml
# CUDA_VISIBLE_DEVICES=1 python train_new.py --config policy_new_tmp.yml

# CUDA_VISIBLE_DEVICES=1 python train.py --config policies/policy_1.yml
# CUDA_VISIBLE_DEVICES=1 python train.py --config policies/policy_1.yml
CUDA_VISIBLE_DEVICES=1 python train.py --config policies/policy_2.yml
CUDA_VISIBLE_DEVICES=1 python train.py --config policies/policy_2.yml
