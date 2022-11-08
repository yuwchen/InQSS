import os
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from train import MosPredictor, MyDataset
import numpy as np
import scipy.stats
import torchaudio
import gc


gc.collect()
torch.cuda.empty_cache()

def systemID(uttID):
    return uttID.split('-')[0]

def get_filepaths(directory):
      file_paths = []  
      for root, directories, files in os.walk(directory):
            for filename in files:
                  # Join the two strings in order to form the full filepath.
                  filepath = os.path.join(root, filename)
                  if filename.endswith('.wav'):
                        file_paths.append(filepath)  
      return file_paths 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fairseq_base_model', type=str, default='fairseq/wav2vec_small.pt', help='Path to pretrained fairseq base model.')
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to finetuned MOS-Intell prediction checkpoint.')
    args = parser.parse_args()
    
    cp_path = args.fairseq_base_model
    my_checkpoint = args.ckpt
    datadir = args.datadir
    
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)
    model.eval()
    model.load_state_dict(torch.load(my_checkpoint))
    
    print('Loading data')
    validset = get_filepaths(datadir)

    print('Starting prediction')
    outfile = my_checkpoint.split("/")[-2]
    output_dir = 'Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ans_mos = open(os.path.join(output_dir, outfile+'_mos.txt'), 'w')
    ans_intell = open(os.path.join(output_dir, outfile+'_intell.txt'), 'w')

    for filepath in validset:
        
        with torch.no_grad():
            filename = filepath.split("/")[-1]
            wav = torchaudio.load(filepath)[0]  
            wav = wav.to(device)
            outputs_mos, outputs_intell = model(wav)
        
            outputs_mos = outputs_mos.cpu().detach().numpy()[0]
            outputs_intell = outputs_intell.cpu().detach().numpy()[0]
            
            #predictions[filename] = output  ## batch size = 1
            print(filename, outputs_mos, outputs_intell)
            
            #del wav, output
            torch.cuda.empty_cache()
            ans_mos.write(filename+" "+str(outputs_mos)+"\n")
            ans_intell.write(filename+" "+str(outputs_intell)+"\n")

    

if __name__ == '__main__':
    main()
