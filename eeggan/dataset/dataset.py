import numpy as np
import pandas as pd
import os


class LazyPerPersonEEGDataset:
    def __init__(self, folderpath = r'../../data/binary'):
        self.folderpath = folderpath
        self.files = []
        for file in os.listdir(folderpath):
                    if file.endswith('.npy'):
                        try:
                            self.files.append(os.path.join(folderpath, file))
                        except FileNotFoundError as e:
                            print(file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> np.ndarray:
        filepath = self.files[index]
        data = np.load(file=filepath)

        return data


class ProcessedEEGDataset:
    def __init__(self, folderpath = r'../../data/binary'):
        self.folderpath = folderpath
        self.files = []
        self.csvs = []


        for file in os.listdir(folderpath):
            try:
                if file.endswith('.npy'):
                    self.files.append(np.load(file=os.path.join(folderpath, file)))
                elif file.endswith('.csv'):
                    self.csvs.append(pd.read_csv(os.path.join(folderpath,file)))
            except FileNotFoundError as e:
                print(file)



    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> np.ndarray:
        data = self.files[index]
        return data
