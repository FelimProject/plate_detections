import pickle
import os
def save_stub_file(stub_path , detections):
    
    with open(stub_path , 'wb') as f:
        pickle.dump(detections , f)

def read_stub_path(stub_path):
    if os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            return pickle.load(f)
    else:
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)  
        with open(stub_path, 'wb') as f:
            pickle.dump({}, f)
