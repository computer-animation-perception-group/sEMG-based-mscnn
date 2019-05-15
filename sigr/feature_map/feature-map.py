import numpy as np
import emg_features

def feature_map(x, feature_list):

    res = []
    for i in range(x.shape[0]):
        single_channel = []
        for j in range(len(feature_list)):
            func = 'emg_features.emg_'+feature_list[j]
            single_channel.append(eval(str(func))(x[i,:]))
        # print single_channel[0].shape
        single_channel = np.hstack(single_channel)
        res.append(single_channel)
    res =np.vstack(res)
    return res