from tqdm import tqdm
from typing import *
from mne.preprocessing import ICA
import pandas as pd
import numpy as np
import numpy.typing as npt
import pymatreader
import mne
import os
import warnings
warnings.filterwarnings('ignore')


class EEGpreprocessing:
    """Processes raw signal.

    Takes eeglab files, performs filtering, Independent Component
    Analysis(ICA), grabs events and then saves binary .npy files.

    NOTE: If you work with cloned repo there is no need to
    pass directory to data.
    """

    def __init__(self, path=r'../../data/raw',output_path = r'../../data/binary'):
        self.files = []
        self.file_name = []
        self.path = path
        self.output_path = output_path
        for file in tqdm(os.listdir(path)):
            if file.endswith('.set'):
                try:
                    self.files.append(os.path.join(path, file))
                    self.file_name.append(file[:-4])

                except FileNotFoundError as e:
                    print(file)

    def _filtering(self, signal: npt.ArrayLike) -> np.ndarray:
        """
        Performs filtering by applying butterworth and
        notch filter.

        Args:
            signal ArrayLike: Raw EEG signal.

        Returns:
            np.ndarray: Filtered signal.
        """
        
        sf = signal.copy().filter(
            l_freq=0.1,
            h_freq=45.0,
            picks=None,
            filter_length='auto',
            l_trans_bandwidth='auto',
            h_trans_bandwidth='auto',
            n_jobs=1, method='iir',
            iir_params={'order':2, 'ftype':'butter'},
            skip_by_annotation=('edge', 'bad_acq_skip'),
            pad='reflect_limited',
            verbose=False
            )

        sfilt = sf.copy().notch_filter(
            freqs = 50.0,
            picks=None, 
            filter_length='auto', 
            notch_widths=None, 
            trans_bandwidth=1.0, 
            n_jobs=1, 
            method='iir', 
            iir_params=None, 
            mt_bandwidth=None, 
            p_value=0.05, 
            pad='reflect_limited', 
            verbose=False
            )   

        return sfilt
        
    def _ica(self, signal: npt.ArrayLike) -> npt.ArrayLike:
        """Applies fastICA to signal.

        Filter out EOG artefacts from the signal.

        Args:
            signal ArrayLike: Filtered signal.

        Returns:
            ArrayLike: Post ICA signal with deleted Diod and EOG channel.
        """
        signal.drop_channels('dioda')
        ica = ICA(n_components=19, method='fastica')
        ica.fit(signal, decim=None, reject={'eeg':10e-4}, verbose=False)
        ica.exclude = []
        eog_indices, eog_scores  = ica.find_bads_eog(
                                                    signal.copy(),
                                                    ch_name='EOG'
                                                    )
        ica.exclude = eog_indices
        eeg_ica  = ica.apply(signal.copy(), exclude = eog_indices)
        eeg_ica.drop_channels('EOG')
        eeg_ica.drop_channels('TSS')
        eeg_ica.drop_channels('A1')
        eeg_ica.drop_channels('A2')

        return eeg_ica
    
    def _setthreshold(self, signal:np.ndarray, threshold: float) -> np.ndarray:
        """Sets threshold and deletes outliers.

        Args:
            signal (np.ndarray): 3 dimentional signal with 0th dim as events. 

        Returns:
            np.ndarray: Returns EEG signal with deleted events. 
        """
        idx = []
        for i in range(len(signal)):
            if np.max(np.abs(signal[i,:,:])) > threshold:
                idx.append(i)
        new_signal = np.delete(signal, idx, axis=0)
                
        return new_signal


    def _read_tag(self, fname):
        from mne.utils._bunch import Bunch
        eeg = pymatreader.read_mat(fname, uint16_codec=None)
        eeg = eeg.get('EEG', eeg)
        eeg = Bunch(**eeg)
        tags_type = eeg.event['tag_type']
        event_type = eeg.event['type']

        return tags_type, event_type


    def prepare(self, threshold = 50.0, Fs = 768, exp_trial_tag = 1,start_at = 0) -> None:
        """Gets events from preprocessed signal.
        
        Combines all processing methods and outputs binary files.
        Shape of saved file is Events x Channels x 256 samples (Fs=256Hz)

        """
        os.makedirs(self.output_path, exist_ok = True)

        for (file,name) in tqdm([m for m in zip(self.files, self.file_name)][start_at:] ):
            if name + '.npy' not in os.listdir(self.output_path):
                raw = mne.io.read_raw_eeglab(file, eog='auto', preload=True)
                mne.set_bipolar_reference(raw, 'A1', 'A2')
                Fs=raw.info['sfreq']
                Fs_int = int(Fs)

                raw.load_data()
                sf = self._filtering(signal=raw)
                s_ica = self._ica(signal=sf)
                n_chan = s_ica.info['nchan']

                eventstarts = mne.events_from_annotations(raw.copy())[0]
                eventstarts = eventstarts[np.where(eventstarts[:,2] == exp_trial_tag)][:,0]
                events = np.zeros((len(eventstarts), n_chan, Fs_int))
                print(events.shape)
                for i, start in enumerate(eventstarts):

                    crop_data = s_ica.copy().crop(
                                            tmin=(start/Fs)-0.2,
                                            tmax=(start/Fs)+0.8,
                                            include_tmax=False
                                            )
                    crop_data_values = crop_data.get_data(units='uV')
                    init_mean = crop_data_values[:int(0.2 * Fs_int)].mean()
                    events[i,:,:] = crop_data_values - init_mean

                clean_signal = self._setthreshold(signal=events, threshold = threshold)
                np.save(file=f'{self.output_path}/{name}.npy', arr=clean_signal)

                # Save csv for each subject
                tags_type, event_type = self._read_tag(file)
                df = pd.DataFrame(tags_type)
                df = df.loc[df.block!=0,:]
                df.to_csv(f'{self.output_path}/{name}_tags.csv',encoding='utf-8')

if __name__ == "__main__":

    data = EEGpreprocessing(path = 'C:/data/eegan',output_path = 'C:/data/eegan/binary')
    data.prepare()
