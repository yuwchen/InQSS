# InQSS: a speech intelligibility and quality assessment model using a multi-task learning network


This code is related to the paper accepted by INTERSPEECH 2022:    
InQSS: a speech intelligibility and quality assessment model using a multi-task learning network. 
https://arxiv.org/abs/2111.02585


## TMHINT-QI dataset

The dataset is publicly available now!  

[Download link](http://gofile.me/6PGhz/4U6GWaOtY) pw:tmhintqi  

train: the wave files of the training utterances  
test: the wave files of the testing utterances  
raw_data.csv: record the corresponding subject index, wave file name, quality score, and intelligibility score  

----
Due to the page limit, some details about the dataset are not written in the paper.

### - How much does the headphone affect the score?

In our test, there are **110** subjects used the provided headphone and **116** subjects used their own headphones.  
The following histograms compare the results between using the provided headphone and using their headphones.  
(Note: all subjects used the provided headphone used the same headphone. )  

<img src="https://github.com/yuwchen/InQSS/blob/main/plot/headphone_avg_quality_snr.png" 
alt="main"  width=40% height=40% />   <img src="https://github.com/yuwchen/InQSS/blob/main/plot/headphone_avg_intelligibility_snr.png" 
alt="main"  width=40% height=40% /> 

-> the average scores under different SNR seem not very much affected by the headphones. 

<img src="https://github.com/yuwchen/InQSS/blob/main/plot/headphone_std_quality_snr.png" 
alt="main"  width=40% height=40% />   <img src="https://github.com/yuwchen/InQSS/blob/main/plot/headphone_std_intell_snr.png" 
alt="main"  width=40% height=40% /> 

-> the standard deviations of using the provided headphone are not necessarily more minor than using their headphones.


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



## License
* The InQSS work is released under MIT License. See LICENSE for more details.

## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan

