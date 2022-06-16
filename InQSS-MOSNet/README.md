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

### Prepare your data

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

### Training
```
python train.py
```


### Testing
```
python test.py --rootdir /path/to/test/wav --pretrained_model /output_model/inqss.h5
```
* [Download](http://gofile.me/6PGhz/5rTKiG9k8) pretained model. Password:inqssmosnet
* (Note: the results in the paper is the average scores of four models, which are trained with different training and validation splits. Therefore, the scores in the paper are different from the results using this pretrained model.)



### Acknowledgment
* "MOSNet: Deep Learning based Objective Assessment for Voice Conversion" [Github](https://github.com/lochenchou/MOSNet) 
