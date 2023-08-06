# import optuna

# # storage = "mysql+mysqlconnector://root:test1234@10.244.103.144:3306/ddp_database"
# storage = 'postgresql+psycopg2://testUser:testPassword@10.244.244.151:5432/testDB'
# study_summaries = optuna.study.get_all_study_summaries(storage=storage)
# for s in study_summaries:
#     print(s.study_name)
import os   
from tqdm import tqdm
from glob import glob
dirs = glob("/root/exps_adam/Con_DG/single/*")
len(dirs)
from os import path
n=0
for d in tqdm(dirs):
    if path.exists(d+"/checkpoints") == False:
        n+=1 
        # print (d)
        os.system(f'rm -r {d}')
    

