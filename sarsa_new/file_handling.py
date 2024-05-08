import json
import pickle

def store_policy(filename, Q):
    with open(filename, 'wb') as file:
        pickle.dump({'Q_table': Q}, file)

def load_policy(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def store_hyperparameters(filename, hyperparameters):
    with open(filename, 'w') as file:
        json.dump({'hyperparameters': hyperparameters}, file)

def load_hyperparameters(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
        return data['hyperparameters']
