# InQSS: a speech intelligibility and quality assessment model using a multi-task learning network
***
## InQSS-MOSNet

### Environment
* create from conda env
```
conda env create -f /path/to/environment.yml
```
* manually install dependencies
#####tensorflow
#####scipy
⋅⋅⋅*librosa
⋅⋅⋅*kymatio
⋅⋅⋅*tqdm

### prepare your data

1. put the wavefiles in data/train
2. extract features by runing
```
python utils.py
```
 ⋅⋅⋅*the extracted spectrogram will be saved in data/train/bin
 ⋅⋅⋅*the extracted scattering coefficients will be saved in data/train/train_scatter
  
3. create data/training_list.txt
 ⋅⋅⋅*the format shoule be as follows:
```
wavname1.wav,quality_score1,intelligibility_score1
wavname2.wav,quality_score2,intelligibility_score2
wavname3.wav,quality_score3,intelligibility_score3
```
 ⋅⋅⋅*reference: /data/training_list.txt
 ⋅⋅⋅*rescale the intelligibility scores and quality scores into similar scales \\
      the scales of intelligibility and quality scores in our experiments are 0.0-5.0 and 1.0-5.0, respectively

### training
```
python train.py
```


### testing
```
python test.py --rootdir /path/to/test/wav --pretrained_model /output_model/inqss.h5
```

### Acknowledgments
* [MOSNet](https://github.com/lochenchou/MOSNet)

***

## InQSS-SSL



### Acknowledgments
* [SSL](https://github.com/nii-yamagishilab/mos-finetune-ssl)



***
## License
* The InQSS work is released under MIT License. See LICENSE for more details.

## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan

