from models.PL_resnet import * 
from models.PL_MMD_AAE import *
import json, shutil
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.tuner import Tuner


if __name__ == "__main__":
    # torch.set_float32_matmul_precision("medium")
    exp_name = 'default'
    
    seed = 10086
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    override_version = None
    if len(sys.argv) > 2:
        override_version = sys.argv[2]

    print("Running Experiment: ", exp_name)
    config_file_path = f'/root/configs/'+ exp_name + '.json'
    f = open(config_file_path)
        # f = open(f'/root/autoencoder_denoiser/configs_baseline_selection/'+ name + '.json')
        # global config
        
    config = json.load(f)
    # Init our model
    if config['model']['name'] == 'resnet18':
        model = Baseline_Resnet(config)
    if config['model']['name'] == 'MMD_AAE':
        model = MMD_AAE(config)            
    '''callbacks'''
    checkpoint_callback = PL.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_last=True)
    early_stop_callback = EarlyStopping(monitor="val/loss", mode="min")
    # lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    # swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    log_dir = f'/root/exps/' 
    os.makedirs(log_dir, exist_ok=True)


    # Initialize a trainer
    
    # version = None
    name, version = exp_name.split('/')
    if override_version:
        version = override_version  
    os.makedirs(os.path.join(log_dir,name,version), exist_ok=True)   
    shutil.copy2(config_file_path,os.path.join(log_dir,name,version))
    
    trainer = PL.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy = 'auto',
        max_epochs=config['experiment']['num_epochs'],
        # logger=CSVLogger(save_dir=log_dir),
        logger = TensorBoardLogger(save_dir=log_dir, name=name, version=version),
        callbacks=[checkpoint_callback,early_stop_callback],
    )
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, mode="power")
    # model = torch.compile(model, mode="reduce-overhead")
    # Train the model ⚡
    trainer.fit(model)
    if config['test']:
        trainer.test(ckpt_path='best')
    #    trainer.test(model, ckpt_path='/root/exps/resnet/resnet18_v1/logs/lightning_logs/version_2/checkpoints/epoch=22-step=27485.ckpt') 