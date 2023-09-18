from models.PL_resnet import * 
from models.PL_MMD_AAE import *
from models.PL_ConDG import *
from models.PL_rand_con import *
from models.PL_RandConv_kernel import *
from models.PL_SNR import *
# import torch._inductor.config as torch_config
import json, shutil
from models.utils import convert_bn_layers, convert_relu_layers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.strategies import DDPSpawnStrategy
# from pytorch_lightning.tuner import Tuner
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from data_module import DeviceFingerpringDataModule, VLCSDataModule
simple_trainer = True #global variable 


class Objective:
    def __init__(self, datamodule, should_early_stop):
        self.should_early_stop = should_early_stop
        self.datamodule = datamodule
        
    def __call__(self, trial: optuna.trial.Trial) -> float:
        if len(sys.argv) > 1:
            exp_name = sys.argv[1]
        override_name = None
        if len(sys.argv) > 2: # indicates DEBUG mode
            override_name = sys.argv[2]
        name, version = exp_name.split('/')
        if override_name:
            name = "dev"  
            
        config = get_config(exp_name)
        sample_config(trial, config)
        
        if config['model']['name'] == "MMD_AAE":
            n_layers = config['model'].get('adv_layer_nums')
            if n_layers:
                config['model']['hidden_units_size'] = [
                    trial.suggest_int(f"adv_units_l{i+1}", config['model']['adv_hidden_size_range'][0],
                                    config['model']['adv_hidden_size_range'][1]  , log=True) for i in range(n_layers)
                ]  
            
        if config['model']['name'] == "ConDG":
            general_n_layers = config['model'].get('general_layer_nums')
            if general_n_layers and "general_hidden_units_size" not in config['model'].keys():
                config['model']['general_hidden_units_size'] = [
                    trial.suggest_int(f"general_layers_units_l{i+1}", config['model']['general_hidden_size_range'][0],
                                    config['model']['general_hidden_size_range'][1], log=True) for i in range(general_n_layers)
                ]  
            
            con_n_layers = config['model'].get('con_layer_nums')
            if con_n_layers and "con_hidden_units_size" not in config['model'].keys() :
                config['model']['con_hidden_units_size'] = [
                    trial.suggest_int(f"con_layers_units_l{i+1}", config['model']['con_hidden_size_range'][0],
                                    config['model']['con_hidden_size_range'][1], log=True) for i in range(con_n_layers)
                ]  
        
            
        config['use_pretrained'] = "pretrain" in  exp_name
        print("Using pretrained mode: ", config['use_pretrained'], "\n")
        
        datamodule = self.datamodule
        # Init our model
        if config['model']['name'] in ['resnet18', "resnet50"] :
            model = Baseline_Resnet(config)
        elif config['model']['name'] == 'MMD_AAE':
            model = MMD_AAE(config)  
        elif config['model']['name'] == 'ConDG':
            model = ConDG(config, datamodule)      
        elif config['model']['name'] == 'RandConv':
            model = RandConv(config)  
        elif config['model']['name'] == "RandConv_kernel":
            model = RandConv_kernel(config)
        elif config['model']['name'] == "SNR":
            model = SNR(config)
            
        if config['experiment'].get("group_norm"):
            print("doing gn! \n\n\n")
            num_groups = config['experiment']['num_groups']
            model = convert_bn_layers(model, torch.nn.BatchNorm1d, torch.nn.GroupNorm, num_groups = num_groups)      
        if config['experiment'].get("tanh"):
            print("doing tanh! \n\n\n")
            model = convert_relu_layers(model, torch.nn.ReLU, torch.nn.Tanh) 
        
        '''callbacks'''
        if simple_trainer:
            callbacks = []
        else:
            checkpoint_callback = PL.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_last=False, save_weights_only=True)
            lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
            callbacks = [checkpoint_callback, lr_monitor_callback]
        # if "resnet" in exp_name:
       
        if self.should_early_stop:
            # print("should early stop!!!\n\n\n")
            early_stop_callback = EarlyStopping(monitor="val/loss", mode="min", patience=15)
            prune_callback = PyTorchLightningPruningCallback(trial, monitor="val/acc")
            callbacks.append(early_stop_callback)
            callbacks.append(prune_callback)
        # swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
        log_dir = f'/root/exps_new_split/' 
        os.makedirs(log_dir, exist_ok=True)


        # Initialize a trainer
        max_epoch = 20  if len(sys.argv) > 2 else config['experiment']['num_epochs']
        
        if torch.cuda.device_count() < 2:
            strategy = 'auto'  
        elif config['model']['name'] == 'ConDG':
            strategy = 'ddp_spawn_find_unused_parameters_true'
        else:    
            strategy = 'ddp_spawn'
        print("using strategy: ", strategy)
        
        version = version+f"/trail{trial.number}"
        trial.set_user_attr("logging_path",os.path.join(log_dir,name,version))
            #    early_stop_callback, 
                    #    prune_callback
        
        if simple_trainer:
            trainer = PL.Trainer(
                # logger = TensorBoardLogger(save_dir=log_dir, name=name, version=version),
                logger = CSVLogger  (save_dir=log_dir, name=name, version=version),
                # logger = False,
                # enable_checkpointing = False,
                max_epochs = max_epoch,
                callbacks=callbacks,
        )
        else:
            trainer = PL.Trainer(
                accelerator="gpu",
                # devices=1,
                # strategy = strategy,
                max_epochs = max_epoch,
                # logger=False,
                logger=CSVLogger  (save_dir=log_dir, name=name, version=version),
                # logger = TensorBoardLogger(save_dir=log_dir, name=name, version=version),
                callbacks=callbacks,
                # reload_dataloaders_every_n_epochs=1,
                # log_every_n_steps=40
                # profiler="simple",
                # enable_progress_bar = False,
            )

        hyperparameters = config
        if trainer.logger:
            trainer.logger.log_hyperparams(hyperparameters)
        # Train the model ⚡
        trainer.fit(model, datamodule=datamodule)
        # if "resnet" in exp_name:
        if self.should_early_stop:
            prune_callback.check_pruned()
        if config['dataset'].get('img'):    
            return model.best_val_acc   
        else: 
            trainer.test(ckpt_path='best', datamodule=datamodule )  
            return trainer.callback_metrics["test/acc"].item()

def get_config(exp_name):
    config_file_path = f'/root/configs_optuna/'+ exp_name + '.json'
    f = open(config_file_path)        
    config = json.load(f)
    return config

def sample_config(trial, config):
    '''set up optuna's guessing'''    
    for k, v in config.items():
        if type(v) == dict:
            sample_config(trial,v)
        if type(v) == list and v[0] == "optuna": # should change it to optuna guessing
            if type(v[1]) == str:
                config[k] = trial.suggest_categorical(k, v[1:])
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
                # print(f'{k} , {v}\n\n')
                config[k] = trial.suggest_float(k, v[1], v[2], log=log, step=step)
                

def delete_bad_ckpt_callback(study, trial):
    print("current trial.value: ", trial.value)
    print("study.best_value: ",study.best_value)
    path = trial.user_attrs["logging_path"]
    # print("trial logging path: ", path)
    # if  trial.number <= 25 or trial.value< max(study.best_value*0.9, 0.55):
    if  trial.number < 5 or trial.value< study.best_value*0.95:
        os.system(f'rm -r {path}'+"/")
    # if study.best_trial.number != trial.number:
    #     study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])    
        
                               
if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.system('nvidia-smi -L')
    print("cpu count: ", os.cpu_count())
    torch.set_float32_matmul_precision("medium")
    # torch._dynamo.config.suppress_errors = True
    seed = 3407
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)
    
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]

    print("Running Experiment: ", exp_name)
    config = get_config(exp_name)

    should_early_stop = config['experiment'].get('should_early_stop')
    if should_early_stop is None:
        should_early_stop = "resnet" in exp_name or "RandConv" in exp_name
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=12,interval_steps=2) if config.get('should_prune') \
                else optuna.pruners.NopPruner() 
        
   
    storage='postgresql+psycopg2://testUser:testPassword@10.244.84.173:5432/testDB'
    print("creating a new study")
    if  len(sys.argv) > 2:
        if config['dataset'].get('img'):
            study_name = "dev_2d"
        else:
            study_name = "dev"  
    # elif 'resnet' in exp_name:
    #     study_name = exp_name  
    else:
        study_name = exp_name
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True, 
    )
    

    #definining datamodule 
    if config['dataset'].get('img'):
            if config['dataset']['name'] == "VLCS":
                datamodule = VLCSDataModule(config=config)
    else:
            datamodule = DeviceFingerpringDataModule(config = config)
            
    callbacks = [] if simple_trainer else [delete_bad_ckpt_callback]
    n_trials = config.get('n_trials') if config.get('n_trials') else 50
    study.optimize(Objective(datamodule, should_early_stop),
                   n_trials=n_trials, 
                   callbacks=callbacks
                   )
    

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

