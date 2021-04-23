CUDA_VISIBLE_DEVICES=1 python cross_dataset.py --model new_results/vanilla_lr_1e-5/checkpoint/epoch_0016.pth
CUDA_VISIBLE_DEVICES=1 python cross_dataset.py --model new_results/pgd_eps_5e-3_iter_10_lr_1e-5/checkpoint/epoch_0008.pth
CUDA_VISIBLE_DEVICES=1 python cross_dataset.py --model new_results/pgd_eps_5e-3_iter_10_l2norm_1e-2_lr_1e-5/checkpoint/epoch_0011.pth
CUDA_VISIBLE_DEVICES=1 python cross_dataset.py --model new_results/pgd_eps_5e-3_iter_10_lr_1e-5_dropblock/checkpoint/epoch_0011.pth