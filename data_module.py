import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
import random, numpy as np, torch, os, pickle
from glob import glob
# from PIL import Image
import cv2

'''device fingerprint dataset'''
class DeviceFingerpringDataModule(pl.LightningDataModule):
    def __init__(self, config, data_from_pickle=None):
        super().__init__()
        self.data_from_pickle = data_from_pickle
        self.config = config
        self.loader_num_worker = 16#os.cpu_count()//4
        self.parepare_dataset()
        num_source_domains = 3 if self.config['dataset'].get('old_split') else 2
        if self.config['dataset'].get('stacked_dataset'):
            num_source_domains = 1
        self.batch_size = int(config['dataset']['batch_size']) // num_source_domains
        # self.setup()

        
    def parepare_dataset(self) :
        print("preparing dataloader")
        pickleFile = open("/root/dataset/ManyTx.pkl","rb")
        all_info = pickle.load(pickleFile)
        data = all_info['data']
        # data=self.data_from_pickle
        # data = np.load('/root/dataset/all_receiver_data.npy', allow_pickle=True)
        print("pickle file loaded")
        self.domained_data = [],[],[],[]
        self.label_distribution = torch.zeros((len(self.domained_data),len(data))) 
        # print("total length: ", len(data))
        for label in range(len(data)):
            if self.config['dataset'].get("only_n_receiver") and \
                    label == self.config['dataset'].get("only_n_receiver"):
                break
            for i in data[label]:
                for date, j in enumerate(i):
                    for k in j[1]:
                        if self.config['dataset'].get('normalize'):
                            mean, std = k.mean(axis=0), k.std(axis=0)
                            k = (k-mean)/std
                            # print(k.mean(0))
                        self.domained_data[date].append((k.T.astype("float32"),label, date)) 
                        self.label_distribution[date,label]+=1
                if self.config['dataset'].get('single_receiver') :
                    break       
        print("finished preparing dataloader")
        random.seed(12)
        for i in range(4):
            random.shuffle(self.domained_data[i])              
        
    def train_dataloader(self):
        if self.config['dataset'].get('stacked_dataset'):
            dataset = StackedDataset(self.df_data_train)
        else:
            dataset = ConcatDataset(self.df_data_train)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # pin_memory=True,
            persistent_workers=True,
            num_workers=self.loader_num_worker,
        )


    def val_dataloader(self):
        return DataLoader(self.df_data_val, 
                          batch_size=512, 
                          num_workers=self.loader_num_worker,
                        #   pin_memory=True,
                          persistent_workers=True
                          )

    def test_dataloader(self):
        return DataLoader(self.df_data_test, 
                          batch_size=512, 
                          num_workers=self.loader_num_worker,
                        #   persistent_workers=True
                          )
     
    def setup(self, stage = None):
        # print("do a set up")
        if self.config['dataset'].get('old_split'):  
            # 3 + 0.5 + 0.5
            val_size = len(self.domained_data[3])//2
            if stage == "fit":
                # print("spllitting train, val, test, should happen only once")
                self.df_data_train = [self.domained_data[i] for i in range(3)]
                self.df_data_val = self.domained_data[3][:val_size]
            if stage == "test":    
                self.df_data_test = self.domained_data[3][val_size:]
                
        else: 
            # 2 + 1 + 1 
            if stage == "fit":
                self.df_data_train = [self.domained_data[i] for i in range(2)]
                self.df_data_val = self.domained_data[2]
            if stage == "test":  
                self.df_data_test = self.domained_data[3]
                

class VLCSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        # self.data_from_pickle = data_from_pickle
        self.config = config
        # self.loader_num_worker = min(os.cpu_count(),64)
        self.loader_num_worker = 16
        self.batch_size = int(config['dataset']['batch_size']) // 3
        self.V_data = ImageDataset("VOC2007", 0)
        self.L_data = ImageDataset("LabelMe", 1)
        self.C_data = ImageDataset("Caltech101", 2)
        self.S_data = ImageDataset("SUN09", 3)
        
        self.label_distribution = torch.zeros(4,5) 
        for iter, dataset in enumerate([self.V_data, self.L_data, self.C_data,self.S_data]):
            self.label_distribution[iter] = dataset.label_distribution
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            ConcatDataset([self.V_data, self.L_data, self.C_data]),
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.loader_num_worker,
            persistent_workers=True
        )
        
    def val_dataloader(self):
        return DataLoader(self.S_data, 
                          batch_size=self.batch_size * 3, 
                          num_workers=self.loader_num_worker,
                          persistent_workers=True,
                          pin_memory=True,
                          )

        
        
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, name, domain_label, verbose=False):
        classes = sorted(glob(f"/root/dataset/VLCS/{name}/*/"))
        self.all_items = []
        self.label_distribution = torch.zeros(5) 
        for class_label, class_path in enumerate(classes):
            print(class_path)
            all_paths = glob(f"{class_path}*")
            self.all_items += [ (all_paths[i], class_label, domain_label) for i in range(len(all_paths))] 
            # for img_path in  all_paths:
            #     img = cv2.imread(img_path)
            #     img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
            #     img = np.asarray(img, dtype="float32")
            #     img = np.transpose(img, (2, 0, 1))
            #     self.all_items.append((img, class_label, domain_label))
            self.label_distribution[class_label] = len(all_paths)
        random.shuffle(self.all_items)
        # self.get_num = 0
        # self.verbose = verbose
    def __getitem__(self, i):
        item = self.all_items[i]
        img = cv2.imread(item[0])
        img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        # if self.verbose:
        #     print(i)
        #     self.get_num+=1
        return img, item[1], item[2]
     
    def __len__(self):
        return len(self.all_items)
    
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

class StackedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.dataset = sum(datasets, [])

    def __getitem__(self, i):
        return self.dataset[i]
        
        # print(len(combined))
        # return zip(*combined)
        # unzipped_data, unzipped_label, unzipped_date = zip(*combined)
        # return np.concatenate(unzipped_data), np.array(unzipped_label), np.array(unzipped_date)

    def __len__(self):
        return len(self.dataset)
