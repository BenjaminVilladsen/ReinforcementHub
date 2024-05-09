import json
import pickle

def store_policy(filename, Q, settings):
    with open(filename, 'wb') as file:
        pickle.dump({'Q_table': Q}, file)

    settings_filename = filename.replace(".pkl", ".json")
    with open(settings_filename, 'w') as file:
        json.dump({'settings': settings}, file)

def load_policy(filename):
    loaded_Q = None
    loaded_settings = None
    with open(filename, 'rb') as file:
        loaded_Q_raw = pickle.load(file)
        loaded_Q = loaded_Q_raw['Q_table']

    filename = filename.replace(".pkl", ".json")
    with open(filename, 'r') as file:
        data = json.load(file)
        loaded_settings = data['settings']

    return loaded_Q, loaded_settings



