import numpy as np

def encode_obs(obs):
    '''
    One-hot encode all data in a CoupEnv observation
    except for coin count

    obs: Observation from CoupEnv.get_obs()

    Return np array
    '''
    def create_and_encode(l, ind):
        a = [0] * l
        if ind != -1:
            a[ind] = 1
        return a

    arr = list()
    for i in range(8):
        arr += create_and_encode(5, obs[i])
    for i in range(8, 16):
        arr += create_and_encode(2, obs[i])
    arr += obs[16:18]
    for i in range(18, 20):
        arr += create_and_encode(32, obs[i])
    arr.append(obs[20])

    return np.array(arr, dtype='int8')