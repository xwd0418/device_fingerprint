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
        self.loader_num_worker = os.cpu_count()
        self.parepare_dataset()
        self.batch_size = int(config['dataset']['batch_size']) // 3
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
        for label in range(len(data)):
            for i in data[label]:
                for date, j in enumerate(i):
                    for k in j[1]:
                        self.domained_data[date].append((k.T.astype("float32"),label, date)) 
                        self.label_distribution[date,label]+=1
                if self.config['dataset'].get('single_receiver') :
                    break       
        print("finished preparing dataloader")
        random.seed(12)
        for i in range(4):
            random.shuffle(self.domained_data[i])
        
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
        return DataLoader(
            ConcatDataset(self.df_data_train),
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.loader_num_worker,
            # persistent_workers=True
        )


    def val_dataloader(self):
        return DataLoader(self.df_data_val, 
                          batch_size=128, 
                          num_workers=self.loader_num_worker,
                        #   persistent_workers=True
                          )

    def test_dataloader(self):
        return DataLoader(self.df_data_test, 
                          batch_size=128, 
                        #   num_workers=self.loader_num_worker,
                        #   persistent_workers=True
                          )
    
    # def setup(self, stage = None):
    #     if stage == "fit":

    #         print("spllitting train, val, test, should happen only once")  
    #         self.df_data_train, self.df_data_val = [],[]
    #         for i in range(3):
    #             val_num = len(self.domained_data[i])//10
    #             t,v = random_split(self.domained_data[i], [len(self.domained_data[i])-val_num,val_num])
    #             self.df_data_train.append(t)
    #             self.df_data_val.append(v)
    #     if stage == "test":    
    #         self.df_data_test = self.domained_data[3]
     
    def setup(self, stage = None):
        val_size = len(self.domained_data[3])//2
        if stage == "fit":
            print("spllitting train, val, test, should happen only once")  
            self.df_data_train = [self.domained_data[i] for i in range(3)]
            self.df_data_val = self.domained_data[3][:val_size]
            # for i in range(3):
            #     t,v = random_split(self.domained_data[i], [len(self.domained_data[i])-val_num,val_num])
            #     self.df_data_train.append(t)
            #     self.df_data_val.append(v)
            
        if stage == "test":    
            self.df_data_test = self.domained_data[3][val_size:]
   
class VLCSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        # self.data_from_pickle = data_from_pickle
        self.config = config
        self.loader_num_worker = min(os.cpu_count(),64)
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

