from models.PL_resnet import * 
from models.PL_MMD_AAE import *
import json, shutil
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.tuner import Tuner
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from data_module import DeviceFingerpringDataModule

def objective(trial: optuna.trial.Trial) -> float:
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    override_name = None
    if len(sys.argv) > 2:
        override_name = sys.argv[2]
    # version = None
    name, version = exp_name.split('/')
    # name=name+"_new_loader"
    if override_name:
        name = override_name  
    # os.makedirs(os.path.join(log_dir,name,version), exist_ok=True)   
    # shutil.copy2(config_file_path,os.path.join(log_dir,name+"_new_loader",version))
    config_file_path = f'/root/configs/'+ exp_name + '.json'
    f = open(config_file_path)        
    config = json.load(f)
    

    '''set up optuna's guessing'''    
    for sub_name, sub_cfg in config.items():
        if type(sub_cfg) == dict:
            for k, v in sub_cfg.items():
                if type(v) == list: # should change it to optuna guessing
                    if type(v[0]) == str:
                        sub_cfg[k] = trial.suggest_categorical(sub_name+"/"+k, v)
                    if type(v[0]) == int:
                        log, step = False, 1
                        if len(v) == 3:
                            if v[2] == "log": log = True
                            else: step = v[2]
                        elif len(v)>3: raise Exception("too many arguments for optuna")
                        sub_cfg[k] = trial.suggest_int(sub_name+"/"+k, v[0], v[1], log=log, step=step)
                    if type(v[0]) == float:
                        log, step = False, None
                        if len(v) == 3:
                            if v[2] == "log": log = True
                            else: step = v[2]
                        elif len(v)>3: raise Exception("too many arguments for optuna")
                        sub_cfg[k] = trial.suggest_float(sub_name+"/"+k, v[0], v[1], log=log, step=step)
                    


    # Init our model
    if config['model']['name'] == 'resnet18':
        model = Baseline_Resnet(config)
    if config['model']['name'] == 'MMD_AAE':
        model = MMD_AAE(config)    
        
    datamodule = DeviceFingerpringDataModule(config =config)
    # model.giant_batch_size = len(datamodule.df_data_train)//config['dataset']['batch_size'] < 50
    
        
    '''callbacks'''
    checkpoint_callback = PL.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_last=True)
    early_stop_callback = EarlyStopping(monitor="val/loss", mode="min", patience=20)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    prune_callback = PyTorchLightningPruningCallback(trial, monitor="val/acc")
    # swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    log_dir = f'/root/exps_autotune/' 
    os.makedirs(log_dir, exist_ok=True)


    # Initialize a trainer
    
    trainer = PL.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy = 'ddp_spawn',
        max_epochs=1000,
        # logger=CSVLogger(save_dir=log_dir),
        logger = TensorBoardLogger(save_dir=log_dir, name=name, version=version),
        callbacks=[checkpoint_callback,early_stop_callback, lr_monitor_callback, prune_callback],
        reload_dataloaders_every_n_epochs=1,
        # log_every_n_steps=40
    )
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, mode="power")
    # model = torch.compile(model, mode="reduce-overhead")
    hyperparameters = config
    trainer.logger.log_hyperparams(hyperparameters)
    # Train the model ⚡
    trainer.fit(model, datamodule=datamodule)
    if config['test']:
        trainer.test(model,ckpt_path='best', datamodule=datamodule)
        
    prune_callback.check_pruned()

    if config.get('optimize_test_acc'):
        return trainer.callback_metrics["test/acc"].item()
    return trainer.callback_metrics["val/acc"].item()

                           
if __name__ == "__main__":
    os.system('nvidia-smi -L')
    print("cpu count: ", os.cpu_count())
    # torch.set_float32_matmul_precision("medium")
    
    seed = 3407
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

    print("Running Experiment: ", exp_name)
    config_file_path = f'/root/configs/'+ exp_name + '.json'
    f = open(config_file_path)        
    config = json.load(f)
            
    pruner = optuna.pruners.NopPruner() if config.get('no_prune') else optuna.pruners.MedianPruner()

    storage = f"sqlite:///Database/{exp_name.split('/')[-1]}.db"
    print("creating a new study")
    study = optuna.create_study(
        study_name=exp_name,
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True, 
    )
    study.optimize(objective, n_trials=100)
    

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
