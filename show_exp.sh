# pip install psycopg2-binary
# apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# storage="mysql+mysqlconnector://root:test1234@10.244.218.242:3306/ddp_database"

storage=postgresql+psycopg2://testUser:testPassword@10.244.98.130:5432/testDB 
optuna-dashboard $storage $1