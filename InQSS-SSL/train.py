
import os
import argparse
import fairseq
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
from tqdm import tqdm


random.seed(1984)

class MosPredictor(nn.Module):
    
    def __init__(self, ssl_model, ssl_out_dim):
        super(MosPredictor, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.output_layer_mos = nn.Linear(self.ssl_features, 1)
        self.output_layer_intell = nn.Linear(self.ssl_features, 1)
        
    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        x_mos = self.output_layer_mos(x)
        x_intell = self.output_layer_intell(x)
        
        return x_mos.squeeze(1), x_intell.squeeze(1)


    
class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_lookup = { }
        self.intell_lookup = {}
        f = open(mos_list, 'r').read().splitlines()

        for line in f:
            parts = line.split(',')
            wavname = parts[0]
            mos = float(parts[1])
            intell = float(parts[2])
            self.mos_lookup[wavname] = mos
            self.intell_lookup[wavname]=intell

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_lookup.keys())

        
    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        score_mos = self.mos_lookup[wavname]
        score_intell = self.intell_lookup[wavname]
        return wav, score_mos, score_intell, wavname
    

    def __len__(self):
        return len(self.wavnames)


    def collate_fn(self, batch):  ## zero padding
        wavs, scores_mos, scores_intell, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)
        scores_mos  = torch.stack([torch.tensor(x) for x in list(scores_mos)], dim=0)
        scores_intell  = torch.stack([torch.tensor(x) for x in list(scores_intell)], dim=0)

        return output_wavs, scores_mos, scores_intell, wavnames

    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='data/train',  type=str, help='Path of your DATA/ directory')
    parser.add_argument('--fairseq_base_model', default='fairseq/wav2vec_small.pt', type=str, help='Path to pretrained fairseq base model')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='checkpoints', help='Output directory for your trained checkpoints')

    args = parser.parse_args()

    cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint = args.finetune_from_checkpoint
    
    if not os.path.exists(ckptdir):
        os.system('mkdir -p ' + ckptdir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, 'train')
    trainlist = os.path.join(datadir, 'training_list.txt')
    validlist = os.path.join(datadir, 'validation_list.txt')

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()
    

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()
   
    trainset = MyDataset(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=trainset.collate_fn)

    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=2, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    net = MosPredictor(ssl_model, SSL_OUT_DIM)
    net = net.to(device)

    if my_checkpoint != None:  ## do (further) finetuning
        net.load_state_dict(torch.load(my_checkpoint))
    
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    PREV_VAL_LOSS=9999999999
    orig_patience=10
    patience=orig_patience
    for epoch in range(1,1001):
        STEPS=0
        net.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader), 0):

            inputs, labels_mos, labels_intell, _ = data
            inputs = inputs.to(device)
            labels_mos = labels_mos.to(device)
            labels_intell = labels_intell.to(device)

            optimizer.zero_grad()
            output_mos, output_intell = net(inputs)
            loss_mos = criterion(output_mos, labels_mos)
            loss_intell = criterion(output_intell, labels_intell)
            loss = loss_mos + loss_intell

            loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += loss.item()
        
        print('EPOCH: ' + str(epoch))
        print('AVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS))
        epoch_val_loss = 0.0
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.memory_allocated()
            torch.cuda.synchronize() 
        
        ## validation
        VALSTEPS=0
        for i, data in enumerate(validloader, 0):
            VALSTEPS+=1
            inputs, labels_mos, labels_intell, filenames = data
            inputs = inputs.to(device)
            labels_mos = labels_mos.to(device)
            labels_intell = labels_intell.to(device)
            
            output_mos, output_intell = net(inputs)
            loss_mos = criterion(output_mos, labels_mos)
            loss_intell = criterion(output_intell, labels_intell)
            loss = loss_mos + loss_intell
            
            epoch_val_loss += loss.item()

        avg_val_loss=epoch_val_loss/VALSTEPS    
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print('Loss has decreased')
            PREV_VAL_LOSS=avg_val_loss
            torch.save(net.state_dict(), os.path.join(ckptdir, 'best'))
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    print('Finished Training')

if __name__ == '__main__':
    main()
