#!/bin/bash

python model.py -f weights_fancy_aug -u 1 -g 1 -a 4 -b 8 -c 55 -v "/Users/gn03249822/Documents/saliency"

#for i in {55..100..5}
#do
#     python model.py -f weights_fancy_aug -u 1 -g 1 -a 1 -b 8 -c $i
#done

#for i in {15..50..5}
#do
#     python model.py -f weights_minMaxPool -n 1 -u 1 -a 1 -b 8 -c $i
#done
#
#for i in {15..50..5}
#do
#     python model.py -f weights_pgd -p 0.01 -u 1 -a 1 -b 8 -c $i
#done
#
#for i in {15..50..5}
#do
#     python model.py -f weights_pgd -p 0.05 -u 1 -a 1 -b 8 -c $i
#done