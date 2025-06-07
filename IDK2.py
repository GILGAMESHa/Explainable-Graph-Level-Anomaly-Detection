import numpy as np
from utils.iNNE_IK import *
from utils.ik_inne_gpu_v1 import *

def idk_kernel_map(list_of_distributions, psi, t=100):
    """
    :param list_of_distributions:
    :param psi:
    :param t:
    :return: idk kernel matrix of shape (n_distributions, n_distributions)
    """

    D_idx = [0]  # index of each distributions
    alldata = []
    n = len(list_of_distributions)
    for i in range(1, n + 1):
        D_idx.append(D_idx[i - 1] + len(list_of_distributions[i - 1]))
        alldata += list_of_distributions[i - 1]
    alldata = np.array(alldata)

    inne_ik = iNN_IK(psi, t)
    all_ikmap = inne_ik.fit_transform(alldata).toarray()

    idkmap = []
    for i in range(n):
        idkmap.append(np.sum(all_ikmap[D_idx[i]:D_idx[i + 1]], axis=0) / (D_idx[i + 1] - D_idx[i]))
    idkmap = np.array(idkmap)

    return idkmap

def idk_square(list_of_distributions, psi1,  psi2, t1=100, t2=100):
    idk_map1 = idk_kernel_map(list_of_distributions, psi1, t1)
    #np.save(idkmapsavepath + "/idkmap1_psi1_"+str(psi1)+".npy", idk_map1)
    inne_ik = iNN_IK(psi2, t2)
    idk_map2 = inne_ik.fit_transform(idk_map1).toarray()
    #np.save(idkmapsavepath + "/idkmap2_psi1_"+str(psi1)+"_psi2_" + str(psi2) + ".npy", idk_map2)
    idkm2_mean = np.average(idk_map2, axis=0) / t1
    idk_score = np.dot(idk_map2, idkm2_mean.T)
    return idk_score

def idk_anomalyDetector(data, psi, t=200):
    inne_ik = IK_inne_gpu(t, psi)
    inne_ik.fit(data)
    idk_map = inne_ik.transform(data)
    idkm_mean = np.average(idk_map, axis=0) / t
    idk_score = np.dot(idk_map, idkm_mean.T)
    return idk_score, idk_map, inne_ik, idkm_mean

# def idk_anomalyDetector2(data_train, data_test, psi, t=100):
#     model = iNN_IK(psi, t)
#     _ = model.fit_transform(data_train).toarray()
#     idk_map = model.transform(data_test).toarray()
#     idkm_mean = np.average(idk_map, axis=0) / t
#     idk_score = np.dot(idk_map, idkm_mean.T)
#     #auc = roc_auc_score(labels, idk_score)
#     return idk_score

def idk_anomalyDetector2(data_train, data_test, psi, t=100):
    model = IK_inne_gpu(t, psi)
    model.fit(data_train)
    idk_map = model.transform(data_test)
    idkm_mean = np.average(idk_map, axis=0) / t
    idk_score = np.dot(idk_map, idkm_mean.T)
    return idk_score

def idk_ad_gpu(data, x_list, psi, t=100):
    inne_ik = IK_inne_gpu(t, psi)
    inne_ik.fit(data)
    idk_map = inne_ik.transform(data)
    # transform the node embedding
    x = np.concatenate(x_list, axis=0)
    x_ik = inne_ik.transform(x)

    shapes = [arr.shape for arr in x_list]
    ik_list = []
    start_idx = 0
    for shape in shapes:
        end_idx = start_idx + shape[0]
        reshaped_arr = x_ik[start_idx:end_idx]
        ik_list.append(reshaped_arr)
        start_idx = end_idx

    idkm_mean = np.mean(idk_map, axis=0) / t

    weight_list = []
    for x in ik_list:
        weight = np.dot(x, idkm_mean.T)
        weight_list.append(weight)

    idk_score = np.dot(idk_map, idkm_mean.T)

    return idk_score, weight_list
