import pickle
import torch
import numpy as np
import torch.nn.functional as F
import os
from scipy.signal import resample


def tuev_subject_id(filename, partition):
    """Recover the official TUEV subject/index from a processed filename.

    Train files use ``<subject>_<record>-<event>.pkl``. Evaluation files use
    ``<label>_<eval_subject>_<segment>-<event>.pkl`` and therefore require the
    second underscore-delimited token, not the event label.
    """
    tokens = os.path.basename(filename).split("_")
    if partition == "train":
        if len(tokens) < 2:
            raise ValueError(f"Unexpected TUEV train filename: {filename}")
        return tokens[0]
    if partition in ("eval", "test"):
        if len(tokens) < 3:
            raise ValueError(f"Unexpected TUEV eval filename: {filename}")
        return tokens[1]
    raise ValueError("partition must be 'train', 'eval', or 'test'")
from scipy.signal import butter, iirnotch, filtfilt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter


class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # from default 200Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


# class CHBMITLoader(torch.utils.data.Dataset):
#     def __init__(self, root, files, sampling_rate=200):
#         self.root = root
#         self.files = files
#         self.default_rate = 256
#         self.sampling_rate = sampling_rate

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, index):
#         print("dir_path:",os.path.join(self.root, self.files[index]))
#         with open(os.path.join(self.root, self.files[index]), "rb") as f: #16,2560
#             sample = pickle.load(f)['X'].astype("float32") 
#         # sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))  /home/dataset/CHB-MIT/clean_segments/train/chb08_10-693760.pkl dir_path: /home/dataset/CHB-MIT/clean_segments/train/chb04_35-2196480.pkl
#         X = sample
#         # 2560 -> 2000, from 256Hz to ?
#         if self.sampling_rate != self.default_rate:
#             X = resample(X, 10 * self.sampling_rate, axis=-1)
#         X = X / (
#             np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
#             + 1e-8
#         )
#         with open(os.path.join(self.root, self.files[index]), "rb") as f: #16,2560
#             sample = pickle.load(f)['y']
#         Y = sample
#         X = torch.FloatTensor(X)
#         return X, Y
class CHBMITLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = os.path.join(self.root, self.files[index])
        with open(file_path, "rb") as handle:
            sample = pickle.load(handle)
        
        X = sample["X"].astype("float32")  # Implementation note.
        # 2560 -> 2000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)  # Implementation note.
        return X, Y

class PTBLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=500):
        self.root = root
        self.files = files
        self.default_rate = 500
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, self.freq * 5, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        # 256 * 5 -> 1000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y

class SEEDLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["data"]
        # 256 * 5 -> 1000, from 256Hz to ?
        # if self.sampling_rate != self.default_rate:
        #     X = resample(X, 5 * self.sampling_rate, axis=-1)
        # X = X / (
        #     np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
        #     + 1e-8
        # )
        Y = int(sample["label"])
        # print("Y:",Y)
        X = torch.FloatTensor(X)
        return X, Y

class SEEDVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        # 256 * 5 -> 1000, from 256Hz to ?
        # if self.sampling_rate != self.default_rate:
        #     X = resample(X, 5 * self.sampling_rate, axis=-1)
        # X = X / (
        #     np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
        #     + 1e-8
        # )
        Y = int(sample["label"])
        # print("Y:",Y)
        X = torch.FloatTensor(X)
        return X, Y

class HARLoader(torch.utils.data.Dataset):
    def __init__(self, dir, list_IDs, sampling_rate=50):
        self.list_IDs = list_IDs
        self.dir = dir
        self.label_map = ["1", "2", "3", "4", "5", "6"]
        self.default_rate = 50
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, "rb"))
        X, y = sample["X"], self.label_map.index(sample["y"])
        if self.sampling_rate != self.default_rate:
            X = resample(X, int(2.56 * self.sampling_rate), axis=-1)
        X = X / (
            np.quantile(
                np.abs(X), q=0.95, interpolation="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        return torch.FloatTensor(X), y


class UnsupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, root_prest, root_shhs):

        # prest dataset
        self.root_prest = root_prest
        exception_files = ["319431_data.npy"]
        self.prest_list = list(
            filter(
                lambda x: ("data" in x) and (x not in exception_files),
                os.listdir(self.root_prest),
            )
        )

        PREST_LENGTH = 2000
        WINDOW_SIZE = 200

        print("(prest) unlabeled data size:", len(self.prest_list) * 16)
        self.prest_idx_all = np.arange(PREST_LENGTH // WINDOW_SIZE)
        self.prest_mask_idx_N = PREST_LENGTH // WINDOW_SIZE // 3

        SHHS_LENGTH = 6000
        # shhs dataset
        self.root_shhs = root_shhs
        self.shhs_list = os.listdir(self.root_shhs)
        print("(shhs) unlabeled data size:", len(self.shhs_list))
        self.shhs_idx_all = np.arange(SHHS_LENGTH // WINDOW_SIZE)
        self.shhs_mask_idx_N = SHHS_LENGTH // WINDOW_SIZE // 5

    def __len__(self):
        return len(self.prest_list) + len(self.shhs_list)

    def prest_load(self, index):
        sample_path = self.prest_list[index]
        # (16, 16, 2000), 10s
        samples = np.load(os.path.join(self.root_prest, sample_path)).astype("float32")

        # find all zeros or all 500 signals and then remove them
        samples_max = np.max(samples, axis=(1, 2))
        samples_min = np.min(samples, axis=(1, 2))
        valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        samples = samples[valid]

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 0

    def shhs_load(self, index):
        sample_path = self.shhs_list[index]
        # (2, 3750) sampled at 125
        sample = pickle.load(open(os.path.join(self.root_shhs, sample_path), "rb"))
        # (2, 6000) resample to 200
        samples = resample(sample, 6000, axis=-1)

        # normalize samples (remove the amplitude)
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        # generate samples and targets and mask_indices
        samples = torch.FloatTensor(samples)

        return samples, 1

    def __getitem__(self, index):
        if index < len(self.prest_list):
            return self.prest_load(index)
        else:
            index = index - len(self.prest_list)
            return self.shhs_load(index)


def collate_fn_unsupervised_pretrain(batch):
    prest_samples, shhs_samples = [], []
    for sample, flag in batch:
        if flag == 0:
            prest_samples.append(sample)
        else:
            shhs_samples.append(sample)

    shhs_samples = torch.stack(shhs_samples, 0)
    if len(prest_samples) > 0:
        prest_samples = torch.cat(prest_samples, 0)
        return prest_samples, shhs_samples
    return 0, shhs_samples

class TUABCHBMITSEEDLoader(torch.utils.data.Dataset): #TUAB,CHB MIT,SEED
    def __init__(self, root_tuab, root_chbmit, root_seed):
        """
Implementation detail.
        
        Implementation note.
Implementation detail.
Implementation detail.
Implementation detail.
        """
        self.root_tuab = root_tuab
        self.tuab_list = []
        for split in ['train', 'val', 'test']:  # Implementation note.
            split_dir = os.path.join(self.root_tuab, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.tuab_list.append(os.path.join(split_dir, f))
        self.tuab_length = 2000  # Implementation note.
        print("(CHB-MIT) unlabeled data size:", len(self.tuab_list))

        # Implementation note.
        self.root_chbmit = root_chbmit
        # Implementation note.
        self.chbmit_list = []
        for split in ['train', 'val', 'test']:  # Implementation note.
            split_dir = os.path.join(self.root_chbmit, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.chbmit_list.append(os.path.join(split_dir, f))
        self.chbmit_length = 2000  # Implementation note.
        print("(CHB-MIT) unlabeled data size:", len(self.chbmit_list))
    
        # Implementation note.
        self.root_seed = root_seed
        self.seed_list = []
        for split in ['train', 'val', 'test']:  # Implementation note.
            split_dir = os.path.join(self.root_seed, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.seed_list.append(os.path.join(split_dir, f))
        self.seed_length = 2000  # Implementation note.
        print("(SEED) unlabeled data size:", len(self.seed_list))
    
        # Implementation note.
        self.window_size = 200  # Implementation note.
        
        # Implementation note.
        self.tuab_idx_all = np.arange(self.tuab_length // self.window_size)
        self.tuab_mask_idx_N = self.tuab_length // self.window_size // 3

        self.chbmit_idx_all = np.arange(self.chbmit_length // self.window_size)
        self.chbmit_mask_idx_N = self.chbmit_length // self.window_size // 3
        
        self.seed_idx_all = np.arange(self.seed_length // self.window_size)
        self.seed_mask_idx_N = self.seed_length // self.window_size // 3
    
    def __len__(self):
        """Implementation note."""
        return len(self.chbmit_list) + len(self.tuab_list) + len(self.seed_list)

    def chbmit_load(self, index):
        # Implementation note.
        sample_path = self.chbmit_list[index]
        # samples = pickle.load(os.path.join(self.root_chbmit, sample_path)).astype("float32")['X'] #16,2560
        with open(os.path.join(self.root_chbmit, sample_path), 'rb') as f: #16,2560
            samples = pickle.load(f)['X'].astype("float32")

        # Implementation note.
        samples = resample(samples, self.chbmit_length, axis=-1)
        
        # Implementation note.
        # samples_max = np.max(samples, axis=(1, 2))
        # samples_min = np.min(samples, axis=(1, 2))
        # valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        # valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        # samples = samples[valid]
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 0  # Implementation note.

    def tuab_load(self, index):

        # Implementation note.
        sample_path = self.tuab_list[index]
        # sample = pickle.load(open(os.path.join(self.root_tuab, sample_path), "rb"))['signal'] #16,1250
        with open(os.path.join(self.root_tuab, sample_path), 'rb') as f: #16,1250
            samples = pickle.load(f)['X'].astype("float32") #.to(dtype=torch.float32)
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 1  # Implementation note.

    def seed_load(self, index):
        # Implementation note.
        sample_path = self.seed_list[index]
        # samples = pickle.load(os.path.join(self.root_seed, sample_path)).astype("float32")['data'] #16,2000
        with open(os.path.join(self.root_seed, sample_path), 'rb') as f: #16,2000
            samples = pickle.load(f)['data'].to(dtype=torch.float32) #.astype("float32")
         
        # Implementation note.
        # samples_max = np.max(samples, axis=(1, 2))
        # samples_min = np.min(samples, axis=(1, 2))
        # valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        # valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        # samples = samples[valid]
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        # samples = torch.FloatTensor(samples)
        return samples, 2  # Implementation note.

    def __getitem__(self, index):
        """Implementation note."""
        if index < len(self.chbmit_list):
            return self.chbmit_load(index)
        elif index < len(self.chbmit_list) + len(self.tuab_list):
            index = index - len(self.chbmit_list)
            return self.tuab_load(index)
        else:
            index = index - len(self.chbmit_list) - len(self.tuab_list)
            return self.seed_load(index)

def collate_fn_tuabchbmitseed_datasets(batch):
    """
    Implementation note.
Implementation detail.
Implementation detail.
    """
    chbmit_samples = []
    tuab_samples = []
    seed_samples = []
    
    # Implementation note.
    for sample, flag in batch:
        if flag == 0:       # CHB-MIT
            chbmit_samples.append(sample)
        elif flag == 1:     # TUAB
            tuab_samples.append(sample)
        else:               # SEED
            seed_samples.append(sample)
    
    # Implementation note.
    def process_samples(samples):
        if len(samples) > 0:
            # Implementation note.
            if samples[0].dim() == 3:  # Implementation note.
                return torch.cat(samples, 0)  # Implementation note.
            else:  # Implementation note.
                return torch.stack(samples, 0)  # Implementation note.
        return []
    
    # Implementation note.
    chbmit_samples = process_samples(chbmit_samples)
    tuab_samples = process_samples(tuab_samples)
    seed_samples= process_samples(seed_samples)
    
    return chbmit_samples, tuab_samples, seed_samples

class TUABTUEVSEEDLoader(torch.utils.data.Dataset): #TUAB,TUEV,SEED
    def __init__(self, root_tuab, root_tuev, root_seed):
        """
Implementation detail.
        
        Implementation note.
Implementation detail.
Implementation detail.
Implementation detail.
        """
        self.root_tuab = root_tuab
        self.tuab_list = []
        for split in ['train', 'val', 'test']:  # Implementation note.
            split_dir = os.path.join(self.root_tuab, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.tuab_list.append(os.path.join(split_dir, f))
        self.tuab_length = 2000  # Implementation note.
        print("(CHB-MIT) unlabeled data size:", len(self.tuab_list))

        # Implementation note.
        self.root_tuev = root_tuev
        self.tuev_list = []
        for split in ['processed_eval','processed_train']:  # Implementation note.
            split_dir = os.path.join(self.root_tuev, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.tuev_list.append(os.path.join(split_dir, f))
        self.tuev_length = 2000  # Implementation note.
        print("(TUEV) unlabeled data size:", len(self.tuev_list))
    
        # Implementation note.
        self.root_seed = root_seed
        self.seed_list = []
        for split in ['train', 'val', 'test']:  # Implementation note.
            split_dir = os.path.join(self.root_seed, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.seed_list.append(os.path.join(split_dir, f))
        self.seed_length = 2000  # Implementation note.
        print("(SEED) unlabeled data size:", len(self.seed_list))
    
        # Implementation note.
        self.window_size = 200  # Implementation note.
        
        # Implementation note.
        self.tuab_idx_all = np.arange(self.tuab_length // self.window_size)
        self.tuab_mask_idx_N = self.tuab_length // self.window_size // 3

        self.tuev_idx_all = np.arange(self.tuev_length // self.window_size)
        self.tuev_mask_idx_N = self.tuev_length // self.window_size // 3
        
        self.seed_idx_all = np.arange(self.seed_length // self.window_size)
        self.seed_mask_idx_N = self.seed_length // self.window_size // 3
    
    def __len__(self):
        """Implementation note."""
        return len(self.tuev_list) + len(self.tuab_list) + len(self.seed_list)

    def tuev_load(self, index):
        # Implementation note.
        sample_path = self.tuev_list[index]
        # sample = pickle.load(open(os.path.join(self.root_tuev, sample_path), "rb"))['signal'] #16,1250
        with open(os.path.join(self.root_tuev, sample_path), 'rb') as f: #16,1250
            samples = pickle.load(f)['signal'].astype("float32") #.to(dtype=torch.float32)
        
        # Implementation note.
        samples = resample(samples, self.tuev_length, axis=-1) 
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 0  # Implementation note.

    def tuab_load(self, index):

        # Implementation note.
        sample_path = self.tuab_list[index]
        # sample = pickle.load(open(os.path.join(self.root_tuab, sample_path), "rb"))['signal'] #16,1250
        with open(os.path.join(self.root_tuab, sample_path), 'rb') as f: #16,1250
            samples = pickle.load(f)['X'].astype("float32") #.to(dtype=torch.float32)
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 1  # Implementation note.

    def seed_load(self, index):
        # Implementation note.
        sample_path = self.seed_list[index]
        # samples = pickle.load(os.path.join(self.root_seed, sample_path)).astype("float32")['data'] #16,2000
        with open(os.path.join(self.root_seed, sample_path), 'rb') as f: #16,2000
            samples = pickle.load(f)['data'].to(dtype=torch.float32) #.astype("float32")
         
        # Implementation note.
        # samples_max = np.max(samples, axis=(1, 2))
        # samples_min = np.min(samples, axis=(1, 2))
        # valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        # valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        # samples = samples[valid]
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        # samples = torch.FloatTensor(samples)
        return samples, 2  # Implementation note.

    def __getitem__(self, index):
        """Implementation note."""
        if index < len(self.tuev_list):
            return self.tuev_load(index)
        elif index < len(self.tuev_list) + len(self.tuab_list):
            index = index - len(self.tuev_list)
            return self.tuab_load(index)
        else:
            index = index - len(self.tuev_list) - len(self.tuab_list)
            return self.seed_load(index)


def collate_fn_tuabtuevseed_datasets(batch):
    """
    Implementation note.
Implementation detail.
Implementation detail.
    """
    tuev_samples = []
    tuab_samples = []
    seed_samples = []
    
    # Implementation note.
    for sample, flag in batch:
        if flag == 0:       # TUEV
            tuev_samples.append(sample)
        elif flag == 1:     # TUAB
            tuab_samples.append(sample)
        else:               # SEED
            seed_samples.append(sample)
    
    # Implementation note.
    def process_samples(samples):
        if len(samples) > 0:
            # Implementation note.
            if samples[0].dim() == 3:  # Implementation note.
                return torch.cat(samples, 0)  # Implementation note.
            else:  # Implementation note.
                return torch.stack(samples, 0)  # Implementation note.
        return []
    
    # Implementation note.
    tuev_samples = process_samples(tuev_samples)
    tuab_samples = process_samples(tuab_samples)
    seed_samples= process_samples(seed_samples)
    
    return tuev_samples, tuab_samples, seed_samples

class TUABTUEVCHBMITLoader(torch.utils.data.Dataset): #TUAB,TUEV,SEED
    def __init__(
        self,
        root_tuab,
        root_tuev,
        root_chbmit,
        source_splits=None,
        target_sample_rate=200,
        target_window_seconds=5,
    ):
        """
Implementation detail.
        
        Implementation note.
Implementation detail.
Implementation detail.
Implementation detail.
        """
        source_splits = source_splits or {
            "tuab": ("train", "val", "test"),
            "tuev": ("processed_eval", "processed_train"),
            "chbmit": ("train", "val", "test"),
        }
        self.target_sample_rate = target_sample_rate
        self.target_window_seconds = target_window_seconds
        self.target_window_samples = target_sample_rate * target_window_seconds

        self.root_tuab = root_tuab
        self.tuab_list = []
        for split in source_splits["tuab"]:
            split_dir = os.path.join(self.root_tuab, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.tuab_list.append(os.path.join(split_dir, f))
        self.tuab_list.sort()
        self.tuab_length = 2000  # Implementation note.
        print("(CHB-MIT) unlabeled data size:", len(self.tuab_list))

        # Implementation note.
        self.root_tuev = root_tuev
        self.tuev_list = []
        for split in source_splits["tuev"]:
            split_dir = os.path.join(self.root_tuev, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.tuev_list.append(os.path.join(split_dir, f))
        self.tuev_list.sort()
        self.tuev_length = self.target_window_samples
        print("(TUEV) unlabeled data size:", len(self.tuev_list))
    
        # Implementation note.
        # Implementation note.
        self.root_chbmit = root_chbmit
        # Implementation note.
        self.chbmit_list = []
        for split in source_splits["chbmit"]:
            split_dir = os.path.join(self.root_chbmit, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.chbmit_list.append(os.path.join(split_dir, f))
        self.chbmit_list.sort()
        self.chbmit_length = 2000  # Implementation note.
        print("(CHB-MIT) unlabeled data size:", len(self.chbmit_list))
    
    
        # Implementation note.
        self.window_size = 200  # Implementation note.
        
        # Implementation note.
        self.tuab_idx_all = np.arange(self.tuab_length // self.window_size)
        self.tuab_mask_idx_N = self.tuab_length // self.window_size // 3

        self.tuev_idx_all = np.arange(self.tuev_length // self.window_size)
        self.tuev_mask_idx_N = self.tuev_length // self.window_size // 3
        
        self.chbmit_idx_all = np.arange(self.chbmit_length // self.window_size)
        self.chbmit_mask_idx_N = self.chbmit_length // self.window_size // 3
    
    def __len__(self):
        """Implementation note."""
        return len(self.tuev_list) + len(self.tuab_list) + len(self.chbmit_list)

    def tuev_load(self, index):
        # Implementation note.
        sample_path = self.tuev_list[index]
        # sample = pickle.load(open(os.path.join(self.root_tuev, sample_path), "rb"))['signal'] #16,1250
        with open(os.path.join(self.root_tuev, sample_path), 'rb') as f: #16,1250
            samples = pickle.load(f)['signal'].astype("float32") #.to(dtype=torch.float32)
        
        # Implementation note.
        samples = resample(samples, self.tuev_length, axis=-1) 
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 0  # Implementation note.

    def tuab_load(self, index):

        # Implementation note.
        sample_path = self.tuab_list[index]
        # sample = pickle.load(open(os.path.join(self.root_tuab, sample_path), "rb"))['signal'] #16,1250
        with open(os.path.join(self.root_tuab, sample_path), 'rb') as f: #16,1250
            samples = pickle.load(f)['X'].astype("float32") #.to(dtype=torch.float32)
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 1  # Implementation note.

    def chbmit_load(self, index):
        # Implementation note.
        sample_path = self.chbmit_list[index]
        # samples = pickle.load(os.path.join(self.root_chbmit, sample_path)).astype("float32")['X'] #16,2560
        with open(os.path.join(self.root_chbmit, sample_path), 'rb') as f: #16,2560
            samples = pickle.load(f)['X'].astype("float32")

        # Implementation note.
        samples = resample(samples, self.chbmit_length, axis=-1)
        
        # Implementation note.
        # samples_max = np.max(samples, axis=(1, 2))
        # samples_min = np.min(samples, axis=(1, 2))
        # valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        # valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        # samples = samples[valid]
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 2  # Implementation note.

    def __getitem__(self, index):
        """Implementation note."""
        if index < len(self.tuev_list):
            return self.tuev_load(index)
        elif index < len(self.tuev_list) + len(self.tuab_list):
            index = index - len(self.tuev_list)
            return self.tuab_load(index)
        else:
            index = index - len(self.tuev_list) - len(self.tuab_list)
            return self.chbmit_load(index)

def collate_fn_tuabtuevchbmit_datasets(batch):
    """
    Implementation note.
Implementation detail.
Implementation detail.
    """
    tuev_samples = []
    tuab_samples = []
    chbmit_samples = []
    
    # Implementation note.
    for sample, flag in batch:
        if flag == 0:       # TUEV
            tuev_samples.append(sample)
        elif flag == 1:     # TUAB
            tuab_samples.append(sample)
        else:               # SEED
            chbmit_samples.append(sample)
    
    # Implementation note.
    def process_samples(samples):
        if len(samples) > 0:
            # Implementation note.
            if samples[0].dim() == 3:  # Implementation note.
                return torch.cat(samples, 0)  # Implementation note.
            else:  # Implementation note.
                return torch.stack(samples, 0)  # Implementation note.
        return []
    
    # Implementation note.
    tuev_samples = process_samples(tuev_samples)
    tuab_samples = process_samples(tuab_samples)
    chbmit_samples= process_samples(chbmit_samples)
    
    return tuev_samples, tuab_samples, chbmit_samples


class TwoDatasetPretrainLoader(torch.utils.data.Dataset):
    """Select two source-train datasets and harmonize 5-second windows."""

    VALID_DATASETS = ("tuev", "tuab", "chbmit")

    def __init__(
        self,
        root_tuab,
        root_tuev,
        root_chbmit,
        datasets,
        target_sample_rate=200,
        target_window_seconds=5,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_window_seconds = target_window_seconds
        self.target_window_samples = target_sample_rate * target_window_seconds
        self.base = TUABTUEVCHBMITLoader(
            root_tuab,
            root_tuev,
            root_chbmit,
            source_splits={
                "tuab": ("train",),
                "tuev": ("processed_train",),
                "chbmit": ("train",),
            },
            target_sample_rate=target_sample_rate,
            target_window_seconds=target_window_seconds,
        )
        self.datasets = tuple(datasets)
        if len(self.datasets) != 2 or len(set(self.datasets)) != 2:
            raise ValueError("Exactly two distinct pretraining datasets are required")
        invalid = set(self.datasets) - set(self.VALID_DATASETS)
        if invalid:
            raise ValueError(f"Unsupported pretraining datasets: {sorted(invalid)}")
        self.segments_per_record = {
            "tuev": 1,
            "tuab": 2,
            "chbmit": 2,
        }
        self.base_lengths = {
            "tuev": len(self.base.tuev_list),
            "tuab": len(self.base.tuab_list),
            "chbmit": len(self.base.chbmit_list),
        }
        self.lengths = {
            name: self.base_lengths[name] * self.segments_per_record[name]
            for name in self.VALID_DATASETS
        }
        self.loaders = {
            "tuev": self.base.tuev_load,
            "tuab": self.base.tuab_load,
            "chbmit": self.base.chbmit_load,
        }

    def __len__(self):
        return sum(self.lengths[name] for name in self.datasets)

    def __getitem__(self, index):
        for name in self.datasets:
            length = self.lengths[name]
            if index < length:
                segments = self.segments_per_record[name]
                base_index = index // segments
                segment_index = index % segments
                sample, source_flag = self.loaders[name](base_index)
                start = segment_index * self.target_window_samples
                stop = start + self.target_window_samples
                sample = sample[:, start:stop]
                if sample.shape[-1] != self.target_window_samples:
                    raise RuntimeError(
                        f"{name} produced {sample.shape[-1]} samples; "
                        f"expected {self.target_window_samples}"
                    )
                return sample, source_flag
            index -= length
        raise IndexError(index)


def collate_fn_two_dataset_pretrain(batch):
    """Merge unlabeled, shape-aligned samples into one tensor."""
    return torch.stack([sample for sample, _source_flag in batch], dim=0)


class ThreeDatasetLoader(torch.utils.data.Dataset): #CHBMIT,TUEV,SEED
    def __init__(self, root_chbmit, root_tuev, root_seed):
        """
Implementation detail.
        
        Implementation note.
Implementation detail.
Implementation detail.
Implementation detail.
        """
        # Implementation note.
        self.root_chbmit = root_chbmit
        # Implementation note.
        self.chbmit_list = []
        for split in ['train', 'val', 'test']:  # Implementation note.
            split_dir = os.path.join(self.root_chbmit, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.chbmit_list.append(os.path.join(split_dir, f))
        self.chbmit_length = 2000  # Implementation note.
        print("(CHB-MIT) unlabeled data size:", len(self.chbmit_list))
        
        # Implementation note.
        self.root_tuev = root_tuev
        self.tuev_list = []
        for split in ['processed_eval','processed_train']:  # Implementation note.
            split_dir = os.path.join(self.root_tuev, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.tuev_list.append(os.path.join(split_dir, f))
        self.tuev_length = 2000  # Implementation note.
        print("(TUEV) unlabeled data size:", len(self.tuev_list))
        
        # Implementation note.
        self.root_seed = root_seed
        self.seed_list = []
        for split in ['train', 'val', 'test']:  # Implementation note.
            split_dir = os.path.join(self.root_seed, split)
            if os.path.exists(split_dir):  # Implementation note.
                for f in os.listdir(split_dir):
                    if f.endswith('.pkl'):
                        self.seed_list.append(os.path.join(split_dir, f))
        self.seed_length = 2000  # Implementation note.
        print("(SEED) unlabeled data size:", len(self.seed_list))
        
        # Implementation note.
        self.window_size = 200  # Implementation note.
        
        # Implementation note.
        self.chbmit_idx_all = np.arange(self.chbmit_length // self.window_size)
        self.chbmit_mask_idx_N = self.chbmit_length // self.window_size // 3
        
        self.tuev_idx_all = np.arange(self.tuev_length // self.window_size)
        self.tuev_mask_idx_N = self.tuev_length // self.window_size // 3
        
        self.seed_idx_all = np.arange(self.seed_length // self.window_size)
        self.seed_mask_idx_N = self.seed_length // self.window_size // 3

    def __len__(self):
        """Implementation note."""
        return len(self.chbmit_list) + len(self.tuev_list) + len(self.seed_list)

    def chbmit_load(self, index):
        # Implementation note.
        sample_path = self.chbmit_list[index]
        # samples = pickle.load(os.path.join(self.root_chbmit, sample_path)).astype("float32")['X'] #16,2560
        with open(os.path.join(self.root_chbmit, sample_path), 'rb') as f: #16,2560
            samples = pickle.load(f)['X'].astype("float32")

        # Implementation note.
        samples = resample(samples, self.chbmit_length, axis=-1)
        
        # Implementation note.
        # samples_max = np.max(samples, axis=(1, 2))
        # samples_min = np.min(samples, axis=(1, 2))
        # valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        # valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        # samples = samples[valid]
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 0  # Implementation note.

    def tuev_load(self, index):
        # Implementation note.
        sample_path = self.tuev_list[index]
        # sample = pickle.load(open(os.path.join(self.root_tuev, sample_path), "rb"))['signal'] #16,1250
        with open(os.path.join(self.root_tuev, sample_path), 'rb') as f: #16,1250
            samples = pickle.load(f)['signal'].astype("float32") #.to(dtype=torch.float32)
        
        # Implementation note.
        samples = resample(samples, self.tuev_length, axis=-1) 
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        samples = torch.FloatTensor(samples)
        return samples, 1  # Implementation note.

    def seed_load(self, index):
        # Implementation note.
        sample_path = self.seed_list[index]
        # samples = pickle.load(os.path.join(self.root_seed, sample_path)).astype("float32")['data'] #16,2000
        with open(os.path.join(self.root_seed, sample_path), 'rb') as f: #16,2000
            samples = pickle.load(f)['data'].to(dtype=torch.float32) #.astype("float32")
         
        # Implementation note.
        # samples_max = np.max(samples, axis=(1, 2))
        # samples_min = np.min(samples, axis=(1, 2))
        # valid = np.where((samples_max > 0) & (samples_min < 0))[0]
        # valid = np.random.choice(valid, min(8, len(valid)), replace=False)
        # samples = samples[valid]
        
        # Implementation note.
        samples = samples / (
            np.quantile(
                np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
            )
            + 1e-8
        )
        # samples = torch.FloatTensor(samples)
        return samples, 2  # Implementation note.

    def __getitem__(self, index):
        """Implementation note."""
        if index < len(self.chbmit_list):
            return self.chbmit_load(index)
        elif index < len(self.chbmit_list) + len(self.tuev_list):
            index = index - len(self.chbmit_list)
            return self.tuev_load(index)
        else:
            index = index - len(self.chbmit_list) - len(self.tuev_list)
            return self.seed_load(index)

def collate_fn_three_datasets(batch):
    """
    Implementation note.
Implementation detail.
Implementation detail.
    """
    chbmit_samples = []
    tuev_samples = []
    seed_samples = []
    
    # Implementation note.
    for sample, flag in batch:
        if flag == 0:       # CHB-MIT
            chbmit_samples.append(sample)
        elif flag == 1:     # TUEV
            tuev_samples.append(sample)
        else:               # SEED
            seed_samples.append(sample)
    
    # Implementation note.
    def process_samples(samples):
        if len(samples) > 0:
            # Implementation note.
            if samples[0].dim() == 3:  # Implementation note.
                return torch.cat(samples, 0)  # Implementation note.
            else:  # Implementation note.
                return torch.stack(samples, 0)  # Implementation note.
        return 0
    
    # Implementation note.
    chbmit_samples = process_samples(chbmit_samples)
    tuev_samples = process_samples(tuev_samples)
    seed_samples= process_samples(seed_samples)
    
    return chbmit_samples, tuev_samples, seed_samples

class EEGSupervisedPretrainLoader(torch.utils.data.Dataset):
    def __init__(self, tuev_data, chb_mit_data, iiic_data, tuab_data):
        # for TUEV
        tuev_root, tuev_files = tuev_data
        self.tuev_root = tuev_root
        self.tuev_files = tuev_files
        self.tuev_size = len(self.tuev_files)

        # for CHB-MIT
        chb_mit_root, chb_mit_files = chb_mit_data
        self.chb_mit_root = chb_mit_root
        self.chb_mit_files = chb_mit_files
        self.chb_mit_size = len(self.chb_mit_files)

        # for IIIC seizure
        iiic_x, iiic_y = iiic_data
        self.iiic_x = iiic_x
        self.iiic_y = iiic_y
        self.iiic_size = len(self.iiic_x)

        # for TUAB
        tuab_root, tuab_files = tuab_data
        self.tuab_root = tuab_root
        self.tuab_files = tuab_files
        self.tuab_size = len(self.tuab_files)

    def __len__(self):
        return self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size

    def tuev_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuev_root, self.tuev_files[index]), "rb")
        )
        X = sample["signal"]
        # 256 * 5 -> 1000
        X = resample(X, 1000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y, 0

    def chb_mit_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.chb_mit_root, self.chb_mit_files[index]), "rb")
        )
        X = sample["X"]
        # 2560 -> 2000
        X = resample(X, 2000, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 1

    def iiic_load(self, index):
        data = self.iiic_x[index]
        samples = torch.FloatTensor(data)
        samples = samples / (
            torch.quantile(torch.abs(samples), q=0.95, dim=-1, keepdim=True) + 1e-8
        )
        y = np.argmax(self.iiic_y[index])
        return samples, y, 2

    def tuab_load(self, index):
        sample = pickle.load(
            open(os.path.join(self.tuab_root, self.tuab_files[index]), "rb")
        )
        X = sample["X"]
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y, 3

    def __getitem__(self, index):
        if index < self.tuev_size:
            return self.tuev_load(index)
        elif index < self.tuev_size + self.chb_mit_size:
            index = index - self.tuev_size
            return self.chb_mit_load(index)
        elif index < self.tuev_size + self.chb_mit_size + self.iiic_size:
            index = index - self.tuev_size - self.chb_mit_size
            return self.iiic_load(index)
        elif (
            index < self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size
        ):
            index = index - self.tuev_size - self.chb_mit_size - self.iiic_size
            return self.tuab_load(index)
        else:
            raise ValueError("index out of range")


def collate_fn_supervised_pretrain(batch):
    tuev_samples, tuev_labels = [], []
    iiic_samples, iiic_labels = [], []
    chb_mit_samples, chb_mit_labels = [], []
    tuab_samples, tuab_labels = [], []

    for sample, labels, idx in batch:
        if idx == 0:
            tuev_samples.append(sample)
            tuev_labels.append(labels)
        elif idx == 1:
            iiic_samples.append(sample)
            iiic_labels.append(labels)
        elif idx == 2:
            chb_mit_samples.append(sample)
            chb_mit_labels.append(labels)
        elif idx == 3:
            tuab_samples.append(sample)
            tuab_labels.append(labels)
        else:
            raise ValueError("idx out of range")

    if len(tuev_samples) > 0:
        tuev_samples = torch.stack(tuev_samples)
        tuev_labels = torch.LongTensor(tuev_labels)
    if len(iiic_samples) > 0:
        iiic_samples = torch.stack(iiic_samples)
        iiic_labels = torch.LongTensor(iiic_labels)
    if len(chb_mit_samples) > 0:
        chb_mit_samples = torch.stack(chb_mit_samples)
        chb_mit_labels = torch.LongTensor(chb_mit_labels)
    if len(tuab_samples) > 0:
        tuab_samples = torch.stack(tuab_samples)
        tuab_labels = torch.LongTensor(tuab_labels)

    return (
        (tuev_samples, tuev_labels),
        (iiic_samples, iiic_labels),
        (chb_mit_samples, chb_mit_labels),
        (tuab_samples, tuab_labels),
    )


# define focal loss on binary classification
def focal_loss(y_hat, y, alpha=0.8, gamma=0.7):
    # y_hat: (N, 1)
    # y: (N, 1)
    # alpha: float
    # gamma: float
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    # y_hat = torch.clamp(y_hat, -75, 75)
    p = torch.sigmoid(y_hat)
    loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p**gamma * (
        1 - y
    ) * torch.log(1 - p)
    return loss.mean()


# define binary cross entropy loss
def BCE(y_hat, y):
    # y_hat: (N, 1)
    # y: (N, 1)
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    loss = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )
    return loss.mean()

