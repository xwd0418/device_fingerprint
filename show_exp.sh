pip install psycopg2-binary
# storage="mysql+mysqlconnector://root:test1234@10.244.218.242:3306/ddp_database"

storage=postgresql+psycopg2://testUser:testPassword@10.244.244.151:5432/testDB 
optuna-dashboard $storage $1