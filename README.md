# InQSS: a speech intelligibility and quality assessment model using a multi-task learning network


This code is related to the paper accepted by INTERSPEECH 2022:    
InQSS: a speech intelligibility and quality assessment model using a multi-task learning network    
https://www.isca-speech.org/archive/pdfs/interspeech_2022/chen22i_interspeech.pdf

## TMHINT-QI dataset

The dataset is publicly available now!  


Google Drive (recommended):
[Download link](https://drive.google.com/file/d/1TMDiz6dnS76hxyeAcCQxeSqqEOH4UDN0/view?usp=sharing)

NARS: [Download link](http://gofile.me/6PGhz/4U6GWaOtY) pw:tmhintqi  
(This link is for previewing samples in the dataset. We have been informed of issues when attempting to download all the data at once using this link. )

train: the wave files of the training utterances  
test: the wave files of the testing utterances  
raw_data.csv: record the corresponding subject index, wave file name, quality score, and intelligibility score  
test_scores.csv: objective scores of testing data

Known missing data: the following samples in raw_data don't have the quality scores, thereby not being used in the training and testing set    
'FCN_snr0_street_TMHINT_b2_22_03',   
'Trans_snr2_pink_TMHINT_g4_24_09',   
'DDAE_snr2_pink_TMHINT_b2_22_01',   
'DDAE_snr5_white_TMHINT_g4_24_07'   

(If you encounter any issues while downloading the dataset, please reach out to the author (yc4093@columbia.edu).) 
#### Notes about the file name:
Method None also indicates a clean wave file. If the method is None, these files are used for the pretest and are part of the SE model training data. The method shows None because it’s not indicated in the wave file name. (Wave files are named as {method}\_{snr-level}\_{noise}\_{cleanUtterance})

----
Due to the page limit, some details about the dataset are not written in the paper.

### - How much does the headphone affect the score?

We designed a website for the listening test, and thus the listeners can do the test in the place they like as long as the environment is quiet.
In our test, there are **110** listeners used the provided headphone and **116** listeners used their own headphones. All listeners used the provided headphone used the same headphone. 

The following histograms compare the results between using the provided headphone and using their headphones.    

<img src="https://github.com/yuwchen/InQSS/blob/main/plot/headphone_avg_quality_snr.png" 
alt="main"  width=40% height=40% />   <img src="https://github.com/yuwchen/InQSS/blob/main/plot/headphone_avg_intelligibility_snr.png" 
alt="main"  width=40% height=40% /> 

-> the average scores under different SNR seem not very much affected by the headphones. 

<img src="https://github.com/yuwchen/InQSS/blob/main/plot/headphone_std_quality_snr.png" 
alt="main"  width=40% height=40% />   <img src="https://github.com/yuwchen/InQSS/blob/main/plot/headphone_std_intell_snr.png" 
alt="main"  width=40% height=40% /> 

-> the standard deviations of using the provided headphone are not necessarily less than using their headphones.


### - The correlation between the results and the user features

#### Gender (Female: 132, Male: 94)

<img src="https://github.com/yuwchen/InQSS/blob/main/plot/gender_avg_quality_snr.png" 
alt="main"  width=40% height=40% />   <img src="https://github.com/yuwchen/InQSS/blob/main/plot/gender_avg_intell_snr.png" 
alt="main"  width=40% height=40% /> 

<img src="https://github.com/yuwchen/InQSS/blob/main/plot/gender_std_quality_snr.png" 
alt="main"  width=40% height=40% />   <img src="https://github.com/yuwchen/InQSS/blob/main/plot/gender_std_intell_snr.png" 
alt="main"  width=40% height=40% /> 

#### Age (20-30: 154, 31-40: 42, 41-50: 30)

<img src="https://github.com/yuwchen/InQSS/blob/main/plot/age_avg_quality_snr.png" 
alt="main"  width=40% height=40% />   <img src="https://github.com/yuwchen/InQSS/blob/main/plot/age_avg_intell_snr.png" 
alt="main"  width=40% height=40% /> 

<img src="https://github.com/yuwchen/InQSS/blob/main/plot/age_std_quality_snr.png" 
alt="main"  width=40% height=40% />   <img src="https://github.com/yuwchen/InQSS/blob/main/plot/age_std_intell_snr.png" 
alt="main"  width=40% height=40% /> 

### The original TMHINT corpus

The following paper describes how the TMHINT corpus was designed: 
https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dnclcdr&s=id=%22093NTCN0714003%22.&searchmode=basic#XXX


### Correlation between subjective scores and objective scores

We use [pysepm](https://github.com/schmiph2/pysepm) to calculate the objective quality and intelligibilty scores of TMHINT-QI testing set. (The details can be found in test_scores.csv.) 
The following heatmap shows the correlation between different subjective scores and objective scores, where avg_quality and avg_intell denote the average subjective quality and intelligibility scores, respectively.

<img src="https://github.com/yuwchen/InQSS/blob/main/plot/correlation_scores.png" 
alt="main"  width=80% height=80% /> 


## Citation
If you use the dataset in your research, please cite:  

@inproceedings{chen2022inqss,  
  title     = {{InQSS}: a speech intelligibility and quality assessment model using a multi-task learning network},  
  author    = {Chen, Yu-Wen and Tsao, Yu},  
  booktitle = {Proc. INTERSPEECH 2022}  
}   

Thanks :-)!

## License
* The InQSS work is released under MIT License. See LICENSE for more details.

## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan

