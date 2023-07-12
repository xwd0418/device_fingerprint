from models.PL_resnet import * 
from models.PL_MMD_AAE import *
from models.PL_ConDG import *
import json, shutil
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.strategies import DDPSpawnStrategy
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
        name = "dev"  
    # os.makedirs(os.path.join(log_dir,name,version), exist_ok=True)   
    # shutil.copy2(config_file_path,os.path.join(log_dir,name+"_new_loader",version))
    config_file_path = f'/root/configs_optuna/'+ exp_name + '.json'
    f = open(config_file_path)        
    config = json.load(f)
    

    sample_config(trial, config)
    
    if config['model']['name'] == "MMD_AAE":
        n_layers = config['model']['adv_layer_nums']
        config['model']['hidden_units_size'] = [
            trial.suggest_int(f"adv_units_l{i+1}", config['model']['adv_hidden_size_range'][0],
                            config['model']['adv_hidden_size_range'][1]  , log=True) for i in range(n_layers)
        ]  
       
    if config['model']['name'] == "ConDG":
        general_n_layers = config['model']['general_layer_nums']
        config['model']['general_hidden_units_size'] = [
            trial.suggest_int(f"general_layers_units_l{i+1}", config['model']['general_hidden_size_range'][0],
                            config['model']['general_hidden_size_range'][1], log=True) for i in range(general_n_layers)
        ]  
        
        con_n_layers = config['model']['con_layer_nums']
        config['model']['con_hidden_units_size'] = [
            trial.suggest_int(f"con_layers_units_l{i+1}", config['model']['con_hidden_size_range'][0],
                            config['model']['con_hidden_size_range'][1], log=True) for i in range(con_n_layers)
        ]  
       
        
    datamodule = DeviceFingerpringDataModule(config = config)
    # Init our model
    if config['model']['name'] == 'resnet18':
        model = Baseline_Resnet(config)
    if config['model']['name'] == 'MMD_AAE':
        model = MMD_AAE(config)  
    if config['model']['name'] == 'ConDG':
        model = ConDG(config, datamodule)  
        
    # model.giant_batch_size = len(datamodule.df_data_train)//config['dataset']['batch_size'] < 50
    
        
    '''callbacks'''
    checkpoint_callback = PL.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_last=False, save_weights_only=True)
    early_stop_callback = EarlyStopping(monitor="val/loss", mode="min", patience=10)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    prune_callback = PyTorchLightningPruningCallback(trial, monitor="val/acc")
    # swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
    log_dir = f'/root/exps_autotune/' 
    os.makedirs(log_dir, exist_ok=True)


    # Initialize a trainer
    max_epoch = 2  if len(sys.argv) > 2 else 1000
    # strategy = 'ddp_spawn_find_unused_parameters_true' if torch.cuda.device_count() >= 2 else "auto"
    if torch.cuda.device_count() < 2:
        strategy = 'auto'  
    elif config['model']['name'] == 'ConDG':
        strategy = 'ddp_spawn_find_unused_parameters_true'
    else:    
        strategy = 'ddp_spawn'
    print("using strategy: ", strategy)
    version = version+f"/trail{trial.number}"
    trial.set_user_attr("logging_path",os.path.join(log_dir,name,version))
    trainer = PL.Trainer(
        accelerator="gpu",
        # devices=1,
        devices=torch.cuda.device_count(),
        strategy = strategy,
        max_epochs = max_epoch,
        logger=CSVLogger          (save_dir=log_dir, name=name, version=version),
        # logger = TensorBoardLogger(save_dir=log_dir, name=name, version=version),
        callbacks=[early_stop_callback, lr_monitor_callback, prune_callback],
        # reload_dataloaders_every_n_epochs=1,
        # log_every_n_steps=40
    )
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, mode="power")
    # model = torch.compile(model, mode="reduce-overhead")
    hyperparameters = config
    trainer.logger.log_hyperparams(hyperparameters)
    # Train the model ⚡
    trainer.fit(model, datamodule=datamodule)
    # print("trainer.callback_metrics ",trainer.callback_metrics)   
    prune_callback.check_pruned()
    trainer.test(ckpt_path='best', datamodule=datamodule )  
    return trainer.callback_metrics["test/acc"].item()

def sample_config(trial, config):
    '''set up optuna's guessing'''    
    for k, v in config.items():
        if type(v) == dict:
            sample_config(trial,v)
        if type(v) == list and v[0] == "optuna": # should change it to optuna guessing
            if type(v[1]) == str:
                config[k] = trial.suggest_categorical(k, v[2:])
            elif type(v[1]) == int:
                log, step = False, 1
                if type(v[-1]) == str:
                    if v[-1] == "log": 
                        log = True
                    else:
                        assert(v[-1].split("=")[0]=="step")
                        step = int(v[-1].split("=")[1])
                config[k] = trial.suggest_int(k, v[1], v[2], log=log, step=step)
            elif type(v[1]) == float:
                log, step = False, None
                if type(v[-1]) == str:
                    if v[-1] == "log": 
                        log = True
                    else:
                        assert(v[-1].split("=")[0]=="step")
                        step = float(v[-1].split("=")[1])
                config[k] = trial.suggest_float(k, v[1], v[2], log=log, step=step)
                

def delete_bad_ckpt_callback(study, trial):
    print("current trial.value: ", trial.value)
    print("study.best_value: ",study.best_value)
    path = trial.user_attrs["logging_path"]
    # print("trial logging path: ", path)
    if  trial.number <= 25 or trial.value< max(0.95*study.best_value, 0.6):
        os.system(f'rm -r {path}')
    # if study.best_trial.number != trial.number:
    #     study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])    
        
                               
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
    config_file_path = f'/root/configs_optuna/'+ exp_name + '.json'
    f = open(config_file_path)        
    config = json.load(f)
            
    pruner = optuna.pruners.NopPruner() if config.get('no_prune') else optuna.pruners.MedianPruner()

   
    storage = "mysql+mysqlconnector://root:test1234@10.244.103.144:3306/ddp_database"
    print("creating a new study")
    study_name = "dev" if  len(sys.argv) > 2 else exp_name
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True, 
    )
    study.optimize(objective, n_trials=1000, callbacks=[delete_bad_ckpt_callback])
    

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
