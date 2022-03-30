import os
import h5py
import scipy
import librosa
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from os.path import join
from kymatio.numpy import Scattering1D

random.seed(1984)

FS = 16000
FFT_SIZE = 512
SGRAM_DIM = FFT_SIZE // 2 + 1
HOP_LENGTH=256
WIN_LENGTH=512

# dir
DATA_DIR = './data'
AUDIO_DIR = join(DATA_DIR, 'train')
BIN_DIR = join(DATA_DIR, 'bin')
SCAT_DIR = join(DATA_DIR, 'train_scatter')


def get_spectrograms(sound_file, fs=FS, fft_size=FFT_SIZE): 

    # Loading sound file
    y, _ = librosa.load(sound_file, sr=fs) # or set sr to hp.sr.

    # stft. D: (1+n_fft//2, T)
    linear = librosa.stft(y=y,
                     n_fft=fft_size, 
                     hop_length=HOP_LENGTH, 
                     win_length=WIN_LENGTH,
                     window=scipy.signal.hamming,
                     )

    # magnitude spectrogram
    mag = np.abs(linear) #(1+n_fft/2, T)
    feat = np.transpose(mag.astype(np.float32))  

    return feat  


def extract_scatter(file_path, fs=FS):

    y, _ = librosa.load(file_path, sr=fs)

    T = y.shape[-1]
    J = 8
    Q = 8
    scattering = Scattering1D(J, T, Q)
    Sx = scattering(y)
    meta = scattering.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)

    return Sx[order0].T, Sx[order1].T, Sx[order2].T

def read_list(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path

def read(file_path):
    
    data_file = h5py.File(file_path, 'r')
    mag_sgram = np.array(data_file['mag_sgram'][:])
    
    timestep = mag_sgram.shape[0]
    mag_sgram = np.reshape(mag_sgram,(1, timestep, SGRAM_DIM))
    
    return {
        'mag_sgram': mag_sgram,
    }   

def pad(array, reference_shape):
    
    result = np.zeros(reference_shape)
    result[:array.shape[0],:array.shape[1],:array.shape[2]] = array

    return result

def data_generator(file_list, bin_root, batch_size=1):
    
    index=0

    while True:
        
        filename = [file_list[index+x].split(',')[0].split('.')[0] for x in range(batch_size)]
        
        for i in range(len(filename)):
            all_feat = read(join(bin_root,filename[i]+'.h5'))
            sgram = all_feat['mag_sgram'] #shape (1, time, feature)
            
            sgram = (sgram - sgram.min()) / (sgram.max() - sgram.min())

            # the very first feat
            if i == 0:
                feat = sgram
                max_timestep = feat.shape[1]
            else:
                if sgram.shape[1] > feat.shape[1]:
                    # extend all feat in feat
                    ref_shape = [feat.shape[0], sgram.shape[1], feat.shape[2]]
                    feat = pad(feat, ref_shape)
                    feat = np.append(feat, sgram, axis=0)
                elif sgram.shape[1] < feat.shape[1]:
                    # extend sgram to feat.shape[1]
                    ref_shape = [sgram.shape[0], feat.shape[1], feat.shape[2]]
                    sgram = pad(sgram, ref_shape)
                    feat = np.append(feat, sgram, axis=0)
                else:
                    # same timestep, append all
                    feat = np.append(feat, sgram, axis=0)
        
            
        for i in range(len(filename)):
             
            scat1 = np.load(os.path.join(bin_root.replace('bin','train_scatter'), 'order1',filename[i]+'.npy'))
            scat2 = np.load(os.path.join(bin_root.replace('bin','train_scatter'), 'order2',filename[i]+'.npy'))
            scat = np.concatenate((scat1, scat2),axis=1)
            scat = np.reshape(scat, (1, -1, 54+179))
            scat = (scat - scat.min()) / (scat.max() - scat.min())

            # the very first feat
            if i == 0:
                feat_s = scat
                max_timestep = feat_s.shape[1]
            else:
                if scat.shape[1] > feat_s.shape[1]:
                    # extend all feat in feat
                    ref_shape = [feat_s.shape[0], scat.shape[1], feat_s.shape[2]]
                    feat_s = pad(feat_s, ref_shape)
                    feat_s = np.append(feat_s, scat, axis=0)
                elif scat.shape[1] < feat_s.shape[1]:
                    # extend sgram to feat.shape[1]
                    ref_shape = [scat.shape[0], feat_s.shape[1], feat_s.shape[2]]
                    scat = pad(scat, ref_shape)
                    feat_s = np.append(feat_s, scat, axis=0)
                else:
                    # same timestep, append all
                    feat_s = np.append(feat_s, scat, axis=0)
        
        if feat.shape[1]!=feat_s.shape[1]:
            tmp = np.zeros((feat.shape[0],feat.shape[1],54+179))
            tmp[:,:feat_s.shape[1],:] = feat_s
            feat_s = tmp
        
        qua = [float(file_list[x+index].split(',')[1]) for x in range(batch_size)]
        qua = np.asarray(qua).reshape([batch_size])
        frame_qua = np.array([qua[i]*np.ones([feat.shape[1],1]) for i in range(batch_size)])
        
        intell = [float(file_list[x+index].split(',')[2]) for x in range(batch_size)]
        intell = np.asarray(intell).reshape([batch_size])
        frame_intell = np.array([intell[i]*np.ones([feat.shape[1],1]) for i in range(batch_size)])
        
        index += batch_size  
        # ensure next batch won't out of range
        if index+batch_size >= len(file_list):
            index = 0
            random.shuffle(file_list)
   

        yield [feat, feat_s], [qua, frame_qua, intell, frame_intell]
    
            
def extract_features():

    audio_dir = AUDIO_DIR
    output_dir = BIN_DIR
    output_dir_scat = SCAT_DIR

    print('audio dir: {}'.format(audio_dir))
    print('output dir of sprctrogram: {}'.format(output_dir))
    print('output dir of scatter coefficients: {}'.format(SCAT_DIR))
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(SCAT_DIR):
        os.makedirs(join(SCAT_DIR,'order1'))
        os.makedirs(join(SCAT_DIR,'order2'))
            
    # get filenames
    files = []
    for f in os.listdir(audio_dir):
        if f.endswith('.wav'):
            files.append(f.split('.')[0])
    
    print('start STFT, {} files found...'.format(len(files)))
            
    for i in tqdm(range(len(files))):
        
        # set audio/visual file path
        audio_file = join(audio_dir, files[i]+'.wav')
        # spectrogram
        mag = get_spectrograms(audio_file)

        with h5py.File(join(output_dir, '{}.h5'.format(files[i])), 'w') as hf:
            hf.create_dataset('mag_sgram', data=mag)

    print('start scattering transform, {} files found...'.format(len(files)))
    for i in tqdm(range(len(files))):

        audio_file = join(audio_dir, files[i]+'.wav')
        _, feat1, feat2 = extract_scatter(audio_file)
        output_path_1 = os.path.join(SCAT_DIR+'/order1/', files[i])
        np.save(output_path_1, feat1)
        output_path_2 = os.path.join(SCAT_DIR+'/order2/', files[i])
        np.save(output_path_2, feat2)


            
if __name__ == '__main__':

    extract_features()
