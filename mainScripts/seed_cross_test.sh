# CUDA_VISIBLE_DEVICES=1 python cross_dataset.py --model seed_results/pgd_eps_5e-3_iter_10_l2norm_1e-2_lr_1e-5_trial_1/checkpoint/epoch_0014.pth >> seed_cross_test.out
# CUDA_VISIBLE_DEVICES=1 python cross_dataset.py --model seed_results/pgd_eps_5e-3_iter_10_l2norm_1e-2_lr_1e-5_trial_2/checkpoint/epoch_0014.pth >> seed_cross_test.out
CUDA_VISIBLE_DEVICES=1 python cross_dataset.py --model seed_results/policy_lr1e-5_trial_1/checkpoint/epoch_0006.pth >> seed_cross_test_vanilla.out
CUDA_VISIBLE_DEVICES=1 python cross_dataset.py --model seed_results/policy_lr1e-5_trial_2/checkpoint/epoch_0006.pth >> seed_cross_test_vanilla.out