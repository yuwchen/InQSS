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

### Prepare your data

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
  
### Training

* [Download](http://gofile.me/6PGhz/HSNnJMlO7) pretained SSL model, put the model in "./fairseq" dir. Password:inqssssl


```
python train.py
```

### Testing
```
python test.py --datadir /path/to/your/test/wav/dir --ckpt /path/to/your/checkpoint/inqss
```
Use the pretrained models:
* [Download](http://gofile.me/6PGhz/HSNnJMlO7) pretained SSL model, put the model in "./fairseq" dir. Password:inqssssl
* [Download](http://gofile.me/6PGhz/dIUZjJPq1) pretained model. Password:inqssssl
* (Note: the results in the paper is the average scores of four models, which are trained with different training and validation splits. Therefore, the scores in the paper are different from the results using this pretrained model.)

### Acknowledgment
* "Generalization Ability of MOS Prediction Networks" [Github](https://github.com/nii-yamagishilab/mos-finetune-ssl)
