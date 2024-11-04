from pathlib import Path
import numpy as np 
import librosa


datadir = Path('datadir')

################ We use librosa,

def audioFeatures(X, rate=16000):
    
    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=rate, 
                                         n_mfcc=40).T,axis=0).T
    
    #mel-scaled spectrogram:
    mel = np.mean(librosa.feature.melspectrogram(X, 
                                                 sr=rate).T,axis=0).T

    # windowed fts:
    stft = np.abs(librosa.stft(X))

    # chroma:
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, 
                                                 sr=rate).T,axis=0).T



    # spectral contrast:
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, 
                                                         sr=rate).T,axis=0).T

    ####### Using tonnetz increases the no. of parameteres by 69M. So we drop it

    # # tonal centroid features (tonnetz)  
    # tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    #                                           sr=rate).T, axis=0).T
    

    return np.concatenate((mfccs, chroma, mel, contrast), axis=1)



class AudioData():
    def __init__(self, path):
        self.file = path
        self.prepareData()
        
    def to_features(self):
        self.tr_audio = audioFeatures(self.tr_audio)
        self.val_audio = audioFeatures(self.val_audio)
        self.tst_audio = audioFeatures(self.tst_audio)

    def prepareData(self):
        ffile = np.load(self.file)

        #audio file range from -1 to 1
        self.tr_audio = ffile['audioTrs_train']
        self.val_audio = ffile['audioTrs_val']
        self.tst_audio = ffile['audioTrs_test']
        
        



if __name__ == "__main__":
    ddatadir = Path('..')
    a = AudioData(ddatadir / 'audVisIdn.npz')
    a.to_features()
    
    np.save(datadir / 'val_audio_features.npy', a.val_audio)
    np.save(datadir / 'tst_audio_features.npy', a.tst_audio)
    np.save(datadir / 'tr_audio_features.npy',  a.tr_audio)