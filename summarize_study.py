import optuna

# storage = "mysql+mysqlconnector://root:test1234@10.244.103.144:3306/ddp_database"
storage = 'postgresql+psycopg2://testUser:testPassword@10.244.244.151:5432/testDB'
study_summaries = optuna.study.get_all_study_summaries(storage=storage)
for s in study_summaries:
    print(s.study_name)