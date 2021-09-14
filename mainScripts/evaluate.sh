#!/bin/bash

for i in {55..75..5}
do
     python model.py -f weights_pgd -u 1 -p 0.0001 -a 1 -b 8 -c $i
done

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