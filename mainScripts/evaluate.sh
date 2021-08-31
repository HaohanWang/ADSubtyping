#!/bin/bash

for i in {15..50..5}
do
     python model.py -f weights_regular_training -u 1 -a 1 -b 8 -c $i
done

for i in {15..50..5}
do
     python model.py -f weights_minMaxPool -m 1 -u 1 -a 1 -b 8 -c $i
done

for i in {15..50..5}
do
     python model.py -f weights_minMaxPool -p 0.01 -u 1 -a 1 -b 8 -c $i
done

for i in {15..50..5}
do
     python model.py -f weights_minMaxPool -p 0.05 -u 1 -a 1 -b 8 -c $i
done