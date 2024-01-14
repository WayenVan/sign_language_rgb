import lmdb
import sys
sys.path.append('src')
from csi_sign_language.utils.lmdb_tool import store_numpy_array, retrieve_numpy_array
env = lmdb.open('./preprocessed/lmdb_test', map_size=1099511627776)


import numpy as np

a = np.random.rand(2,3)
store_numpy_array(env, 'a', a)
b = retrieve_numpy_array(env, 'a', a.shape, a.dtype)
print(b)
env.close()