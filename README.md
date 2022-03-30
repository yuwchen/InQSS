# InQSS: a speech intelligibility and quality assessment model using a multi-task learning network


This code is related to the paper:    
InQSS: a speech intelligibility and quality assessment model using a multi-task learning network. 
https://arxiv.org/abs/2111.02585


## TMHINT-QI dataset
[download link] (The dataset will be realsed after the paper is formally published)

***
## InQSS-MOSNet

### Environment
* create from conda env
```
conda env create -f /path/to/environment.yml
```
* manually install dependencies 
```
tensorflow 
scipy 
librosa 
kymatio 
tqdm 
```

### prepare your data

1. put the wavefiles in data/train
2. extract features by runing
```
python utils.py
```
* the extracted spectrogram and scattering coefficients will be saved in data/train/bin and data/train/train_scatter, respectively
  
3. create data/training_list.txt
* the format shoule be as follows:
```
wavname1.wav,quality_score1,intelligibility_score1
wavname2.wav,quality_score2,intelligibility_score2
...
```
* reference: /data/training_list.txt
* rescale the intelligibility scores and quality scores into similar scales       
  the scales of intelligibility and quality scores in our experiments are 0.0-5.0 and 1.0-5.0, respectively

### training
```
python train.py
```


### testing
```
python test.py --rootdir /path/to/test/wav --pretrained_model /output_model/inqss.h5
```
* [Download](http://gofile.me/6PGhz/5rTKiG9k8) pretained model. Password:inqssmosnet
* (Note: the results in the paper is the average scores of four models, so the scores in the paper are different from the results using this pretrained model.)

### Acknowledgment
* "MOSNet: Deep Learning based Objective Assessment for Voice Conversion" [Github](https://github.com/lochenchou/MOSNet) 

***

## InQSS-SSL

* create from conda env
```
conda env create -f /path/to/environment.yml
```
* manually install dependencies 
```
pytorch
torchaudio
fairseq 
scipy
tqdm 
```

Note: Recommended to download fairseq from github instead of using pip
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

### prepare your data

1. put the wavefiles in data/train
2. create data/training_list.txt and data/validation_list.txt
* the format shoule be as follows:
```
wavname1.wav,quality_score1,intelligibility_score1
wavname2.wav,quality_score2,intelligibility_score2
...
```
* rescale the intelligibility scores and quality scores into similar scales       
  the scales of intelligibility and quality scores in our experiments are 0.0-5.0 and 1.0-5.0, respectively
  
### training

* [Download](http://gofile.me/6PGhz/HSNnJMlO7) pretained SSL model, put the model in "./fairseq" dir. Password:inqssssl


```
python train.py
```

### testing
```
python test.py --datadir /path/to/your/test/wav/dir --ckpt /path/to/your/checkpoint/inqss
```

* [Download](http://gofile.me/6PGhz/dIUZjJPq1) pretained model. Password:inqssssl
* (Note: the results in the paper is the average scores of four models, so the scores in the paper are different from the results using this pretrained model.)

### Acknowledgment
* "Generalization Ability of MOS Prediction Networks" [Github](https://github.com/nii-yamagishilab/mos-finetune-ssl)

***

## License
* The InQSS work is released under MIT License. See LICENSE for more details.

## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan

