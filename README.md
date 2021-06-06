# Deep Learning for Alzheimer Disease 

## Datasets

Four datasets are available:
  - [ADNI](http://adni.loni.usc.edu/)
  - [AIBL](https://aibl.csiro.au/)
  - [OASIS](https://www.oasis-brains.org/)
  - [MIRIAD](https://www.ucl.ac.uk/drc/research/research-methods/minimal-interval-resonance-imaging-alzheimers-disease-miriad)

ADNI's raw data has been uploaded to our [s3](https://adni-raw-t1w.s3.amazonaws.com) bucket. (official downloading took nearly 1 week to finish! If you want to download other dataset from their official site, chrome should be better than other downloading softwares.)

## Preprocessing

We use [clinica](https://github.com/aramis-lab/clinica) and [AD-DL](https://github.com/aramis-lab/AD-DL) pipelines provided by [ARAMIS Lab](https://github.com/aramis-lab) to run MRI preprocessing. In addition to the original image data, you should also download required clinical data (some csv tables) to run the preprocessing.

Aramis Lab's preprocessing pipeline consisted of two steps: 1. convert the raw data into BIDS format (clinical data is required here); 2. image preprocessing. You can explore it [here](https://camo.githubusercontent.com/6d255d9b1eb580cd81665874330e0a903621d3839640a59aa623e356bd935fb0/687474703a2f2f7777772e636c696e6963612e72756e2f696d672f436c696e6963615f4578616d706c655f323032312d30342d30325f37356470692e6a7067).

[Bin Lu et al.](https://www.biorxiv.org/content/10.1101/2020.08.18.256594v4.full.pdf) recently published a paper on this topic with a different preprocessing pipeline (the main difference with the previous pipeline is that they add segmentation step). You can download its docker image by:
```
docker pull cgyan/brainimagenet
```
However, their pipeline doesn't have a well-written documentation, it might take some time to figure out how it works.

## Models

Deep learning models for 3D images can be too large to fit into one single GPU. We tried using aramis-lab's [model](https://github.com/aramis-lab/AD-DL/blob/6b8f10577663ac5508706eac7cb11319723dbc86/clinicadl/clinicadl/tools/deep_learning/models/image_level.py). Its accuracy is 70-80%. A larger model might reach a higher accuracy. For example, [Bin Lu et al.](https://github.com/Chaogan-Yan/BrainImageNet) claimed their model can reach ~95% accuracy (though we have not reproduced their results).
