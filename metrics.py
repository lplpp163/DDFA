import numpy as np
import skimage.filters as skf
import torch



def mask_mae(pred,gt,mask):
    return np.mean(np.abs(gt[mask]-pred[mask]))
def mask_mse(pred,gt,mask):
    return np.mean(np.power((gt[mask]-pred[mask]),2))

def mask_abs_rel(pred,gt,mask):
    return  np.mean(np.abs(gt[mask]-pred[mask])/(gt[mask]))
def mask_sq_rel(pred,gt,mask):
    return np.mean(np.power((gt[mask]-pred[mask]),2)/(gt[mask]))

def mask_rmse(pred,gt,mask):
    return np.sqrt(np.mean(np.power(pred[mask]-gt[mask],2)))
def mask_rmse_log(pred,gt,mask):
    if np.any(gt[mask] < 0) or np.any(pred[mask] < 0):
        return float('nan')
    gt = np.log(gt[mask])
    pred = np.log(pred[mask])
    out=np.power((gt - pred),2)
    return np.sqrt(np.mean(out))

def mask_accuracy_k(pred,gt,k,mask):
    with np.errstate(divide='ignore'):
        A=pred[mask]/gt[mask]
        B=gt[mask]/pred[mask]
    #A=np.nan_to_num(A)
    #B=np.nan_to_num(B)
    thresh = np.maximum(A , B)
    total_pixels=np.sum(mask)
    #Dp=np.nan_to_num(Dp)
    Dp=np.where(thresh <(1.25**k),1,0)
    return (np.sum(Dp))/total_pixels

#https://github.com/albert100121/AiFDepthNet/blob/master/test.py
def get_bumpiness(gt, algo_result, mask, clip=0.05, factor=100):
    # init
    if type(gt) == torch.Tensor:
        gt = gt.cpu().numpy()[0, 0]
    if type(algo_result) == torch.Tensor:
        algo_result = algo_result.cpu().numpy()[0, 0]
    if type(mask) == torch.Tensor:
        mask = mask.cpu().numpy()[0, 0]
    # Frobenius norm of the Hesse matrix
    diff = np.asarray(algo_result - gt, dtype='float64')
    dx = skf.scharr_v(diff)
    dy = skf.scharr_h(diff)
    dxx = skf.scharr_v(dx)
    dxy = skf.scharr_h(dx)
    dyy = skf.scharr_h(dy)
    dyx = skf.scharr_v(dy)
    bumpiness = np.sqrt(
        np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
    bumpiness = np.clip(bumpiness, 0, clip)
    # return bumpiness
    return np.mean(bumpiness[mask]) * factor