from models.PL_resnet import * 
from models.PL_MMD_AAE import *
import json
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, BatchSizeFinder
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    exp_name = 'default'
    
    seed = 10086
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    sanity_check = None
    if len(sys.argv) > 2:
        sanity_check = sys.argv[2]

    print("Running Experiment: ", exp_name)
    f = open(f'/root/configs/'+ exp_name + '.json')
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
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    batch_size_finder_callback = BatchSizeFinder()
    log_dir = f'/root/exps/' 
    os.makedirs(log_dir, exist_ok=True)


    # Initialize a trainer
    
    name  = exp_name
    version = None
    if sanity_check:
        version = "sanity_check"     
    trainer = PL.Trainer(
        accelerator="cpu",
        # devices=torch.cuda.device_count(),
        # strategy = "ddp",
        max_epochs=config['experiment']['num_epochs'],
        # logger=CSVLogger(save_dir=log_dir),
        logger = TensorBoardLogger(save_dir=log_dir, name=name, version=version),
        callbacks=[checkpoint_callback,early_stop_callback,lr_monitor_callback,],
        auto_scale_batch_size = True
    )

    # Train the model ⚡
    trainer.fit(model,ckpt_path=config['experiment']['ckpt_path'] )
    if config['test']:
        trainer.test()
    #    trainer.test(model, ckpt_path='/root/exps/resnet/resnet18_v1/logs/lightning_logs/version_2/checkpoints/epoch=22-step=27485.ckpt') 