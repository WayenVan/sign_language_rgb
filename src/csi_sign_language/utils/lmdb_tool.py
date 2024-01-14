import lmdb
import numpy as np

def store_numpy_array(env, key, array):
    # Serialize NumPy array to bytes
    array_bytes = array.tobytes()

    # Open a transaction and store the array in the LMDB database
    with env.begin(write=True) as txn:
        txn.put(key.encode('utf-8'), array_bytes)

def retrieve_numpy_array(env, key, shape, dtype):
    # Open a transaction and retrieve the array from the LMDB database
    with env.begin() as txn:
        array_bytes = txn.get(key.encode('utf-8'))

    # Deserialize bytes to NumPy array
    array = np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
    return array