import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def R2(pred, true):
    SS_res = np.sum((true - pred) ** 2)  # 残差平方和
    SS_tot = np.sum((true - true.mean()) ** 2)  # 总平方和
    return 1 - (SS_res / SS_tot)

# Adjusted R² (校正决定系数)
def Adjusted_R2(pred, true, p):
    n = len(true)  # 样本数量
    r2 = R2(pred, true)  # 计算 R²
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

def metric(pred, true,p):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)
    adjr2=Adjusted_R2(pred, true, p)
    return mae,mse,rmse,mape,mspe,r2,adjr2