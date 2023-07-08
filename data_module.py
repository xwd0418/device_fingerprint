import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import random, numpy as np, torch, os


class DeviceFingerpringDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loader_num_worker = os.cpu_count()
        self.parepare_dataset()
        self.batch_size = config['dataset']['batch_size']//3
        # self.setup()

        
    def parepare_dataset(self) :
        print("preparing dataloader")
        # pickleFile = open("/root/dataset/ManyTx.pkl","rb")
        # all_info = pickle.load(pickleFile)
        # data = all_info['data']
        data = np.load('/root/dataset/all_receiver_data.npy', allow_pickle=True)
        print("pickle file loaded")
        self.domained_data = [],[],[],[]
        for label in range(len(data)):
            # one_hot_encoded = F.one_hot(torch.tensor([label]), num_classes=len(data))
            for i in data[label]:
                for date, j in enumerate(i):
                    for k in j[1]: # remove this [0:10] later!!!
                        self.domained_data[date].append((k.T.astype("float32"),label, date)) 
                if self.config['dataset'].get('single_receiver') :
                    break       
        print("finished preparing dataloader")
        
    # def parepare_dataset(self) :
    #     # pickleFile = open("/root/dataset/ManyTx.pkl","rb")
    #     # all_info = pickle.load(pickleFile)
    #     # data = all_info['data']
    #     # data = np.load('/root/dataset/all_receiver_data.npy', allow_pickle=True)
    #     # print("pickle file loaded")
    #     self.domained_data = [[], [], [], []]
    #     loader_mode = "single" if self.config['dataset'].get('single_receiver') \
    #                     else "all"
    #     for i in range(4):
    #         print(f"preparing dataloader in domain_{i}")
    #         loaded = np.load(f"/root/dataset/{loader_mode}_receiver_domain{i}.npz")
    #         data, label, date = loaded['data'], loaded['label'], loaded['date']
    #         self.domained_data[i] = list(zip(data[0], label[0], date[0]))
    #         print(f"finished domain_{i}")               
        
    def train_dataloader(self):
        for d in self.df_data_train:
            random.shuffle(d.indices)
        print("traing loader is shuffled")
        return DataLoader(
            ConcatDataset(self.df_data_train),
            batch_size=self.batch_size,
            pin_memory=True,
            # num_workers=self.loader_num_worker,
            # persistent_workers=True
        )


    def val_dataloader(self):
        return DataLoader(
            ConcatDataset(self.df_data_val),
            batch_size=512//3,
            pin_memory=True,
            # num_workers=self.loader_num_worker,
            # persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(self.df_data_test, 
                          batch_size=4096, 
                        #   num_workers=self.loader_num_worker,
                        #   persistent_workers=True
                          )
    
    def setup(self, stage = None):
        if stage == "fit":

            print("spllitting train, val, test, should happen only once")  
            self.df_data_train, self.df_data_val = [],[]
            for i in range(3):
                val_num = len(self.domained_data[i])//10
                t,v = random_split(self.domained_data[i], [len(self.domained_data[i])-val_num,val_num])
                self.df_data_train.append(t)
                self.df_data_val.append(v)
        if stage == "test":    
            self.df_data_test = self.domained_data[3]
        
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)
        
        # print(len(combined))
        # return zip(*combined)
        # unzipped_data, unzipped_label, unzipped_date = zip(*combined)
        # return np.concatenate(unzipped_data), np.array(unzipped_label), np.array(unzipped_date)

    def __len__(self):
        return min(len(d) for d in self.datasets)                

