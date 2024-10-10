# %%
import sys
import cupy as cp
import numpy as np
import pandas as pd
import time
from lib_gpu import *
from cupy.linalg import norm
import torch

print("Python version:\n", sys.version)
print("Torch version:", torch.__version__)
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("No CUDA GPUs are available, use CPU!")
##############################################################################
# %%
def examples(n, ex, COV, p = None, seed = None):
    m = 0.5
    if seed is not None:
        cp.random.seed(seed=seed)
        
    if ex == 0:
        q,p,d = 1,9,1
        # Define the q sequence
        y = cp.zeros((n, q))
        y[0:p, :] = cp.random.randn(p, q)

        # Define true beta matrix: qp times d.
        B = cp.array([[1/6, 1/6, 1/6, 0, 0, 0, 1/6, 1/6, 1/6],
                      [0, 0, 0, 1/6, -1/6, 1/6, 0, 0, 0]])
        beta = B[:d, :p].T

        # Define Y from times p to T.
        for i in range(p, n, 1):
            x = y[cp.arange(i-1, i-p-1, -1)]

            # Define transformed Xt
            t = beta[:,0] @ x

            # Define each y
            y[i, 0] = t + 0.1 * cp.random.randn(1)
    elif ex == 1:
        q,p,d = 2,4,2
        # Define the q sequence
        y = cp.zeros((n, q))
        y[0:p, :] = cp.random.randn(p, q)

        # Define true beta matrix: qp times d.
        B = cp.array([[1/6, 1/6, 1/6, 0, 0, 0, 1/6, 1/6],
                      [0, 0, 0, 1/6, -1/6, 1/6, 0, 0]])
        beta = B[:d, :(p*q)].T


        ## error setting
        mean = cp.zeros(q)
        cov1 = cp.diag(cp.ones(q))
        cov2 = cp.ones((q,q)) * m + (1-m) * cp.diag(cp.ones(q))
        z = cp.random.randn(n-p, q)
        if COV ==1:
            L = cp.linalg.cholesky(cov1)
        elif COV == 2:
            L = cp.linalg.cholesky(cov2)
        err = mean + z @ L.T

        # Define Y from times p to T.
        for i in range(p, n, 1):
            x = y[cp.arange(i-1,i-p-1,-1),:].reshape(-1)

            # Define transformed Xt
            t1 = beta[:,0] @ x
            t2 = beta[:,1] @ x

            # Define each y
            y[i, 0] = t1 + 0.1 * err[i-p,0]
            y[i, 1] =  t2 + 0.1 * err[i-p,1]         
    elif ex == 2:
        q,p,d = 3,2,2
        # Define the q sequence
        y = cp.zeros((n, q))
        y[0:p, :] = cp.random.randn(p, q)

        # Define true beta matrix: qp times d.
        B = cp.array([[1/6,1/6,1/6,0,0,0], 
                      [0,0,0,1/6,-1/6, 1/6]])
        beta = B[:d, :(p*q)].T

        ## error setting
        mean = cp.zeros(q)
        cov1 = cp.diag(cp.ones(q))
        cov2 = cp.ones((q,q)) * m + (1-m) * cp.diag(cp.ones(q))
        z = cp.random.randn(n-p, q)
        if COV ==1:
            L = cp.linalg.cholesky(cov1)
        elif COV == 2:
            L = cp.linalg.cholesky(cov2)
        err = mean + z @ L.T

        # Define Y from times p to T.
        for i in range(p, n, 1):
            x = y[cp.arange(i-1,i-p-1,-1),:].reshape(-1)

            # Define transformed Xt
            t1 = beta[:,0] @ x
            t2 = beta[:,1] @ x

            # Define each y
            y[i, 0] = cp.exp(t1) + cp.sin(t2) + 0.1 * err[i-p,0]
            y[i,1] = t2/(0.5 + t1**2) + 0.1 * err[i-p,1]
            y[i,2] = cp.abs(t1) * err[i-p,2]    
    elif ex == 3:
        # sparse example q=1
        q, p, d = 1, 24, 1
        # initial the time series
        y = cp.zeros((n,q))
        y[0:p,:] = cp.random.randn(p,q)
        # true beta
        beta = cp.zeros((q*p,d))
        beta[cp.arange(0,q,1),0] = cp.ones(q)*1/6
        beta[cp.arange(q*(p-1), q*p, 1),0] = cp.ones(q)*1/6

        ## error setting
        mean = cp.zeros(q)
        cov1 = cp.diag(cp.ones(q))
        cov2 = cp.ones((q,q)) * m + (1-m) * cp.diag(cp.ones(q))
        z = cp.random.randn(n-p, q)
        if COV ==1:
            L = cp.linalg.cholesky(cov1)
        elif COV == 2:
            L = cp.linalg.cholesky(cov2)
        err = mean + z @ L.T

        for i in range(p,n,1):
            xi = y[cp.arange(i-1,i-p-1,-1),:].reshape(-1)
            t1 = beta[:,0] @ xi
            y[i,0] = cp.tanh(t1) + 0.1 * cp.random.randn(1)
            
    elif ex == 4:
        # Model 2
        q, d = 2, 2
        if p is None:
            p = 12
        # initial the time series
        y = cp.zeros((n,q))
        y[0:p,:] = cp.random.randn(p,q)
        # true beta
        beta = cp.zeros((q*p,d))

        beta[cp.array([0,q*(p-1)]),0] = cp.ones(q)*1/6
        beta[cp.array([1,q*p-1]),1] = cp.ones(q)*1/6

        ## error setting
        mean = cp.zeros(q)
        cov1 = cp.diag(cp.ones(q))
        cov2 = cp.ones((q,q)) * m + (1-m) * cp.diag(cp.ones(q))
        z = cp.random.randn(n-p, q)
        if COV ==1:
            L = cp.linalg.cholesky(cov1)
        elif COV == 2:
            L = cp.linalg.cholesky(cov2)
        err = mean + z @ L.T

        for i in range(p,n,1):
            xi = y[cp.arange(i-1,i-p-1,-1),:].reshape(-1)
            t1 = beta[:,0] @ xi
            t2 = beta[:,1] @ xi

            y[i,0] = t1  + 0.1 * err[i-p,0]
            y[i,1] = cp.tanh(t2) + 0.1 * err[i-p,1]


    elif ex == 5:
        q,p,d = 10,5,3
        # Define the q sequence
        y = cp.zeros((n, q))
        y[0:p, :] = cp.random.randn(p, q)

        # Define true beta matrix: qp times d.
    #     beta = cp.random.rand(q*p, d) - 0.5
        B = cp.zeros((d, p*q))
        B[0, 0:20] = cp.ones(20) * 0.05
        B[1, 20:30] = - cp.ones(10) * 0.1
        B[2, 30:50] = cp.ones(20) * 0.05
        beta = B[:d, :(p*q)].T
        # print(beta)
        A = cp.random.rand(q,d)
        A = cp.zeros((q, d))
        A[:d,:d] = cp.array([[1/2, -1/2, -1], 
                            [-1/2, 1/2, -1/2],
                            [0, -1/2,1/2]])
        
        ## error setting
        mean = cp.zeros(q)
        cov1 = cp.diag(cp.ones(q))
        cov2 = cp.ones((q,q)) * m + (1-m) * cp.diag(cp.ones(q))
        z = cp.random.randn(n-p, q)
        if COV ==1:
            L = cp.linalg.cholesky(cov1)
        elif COV == 2:
            L = cp.linalg.cholesky(cov2)
        err = mean + z @ L.T
        # Define Y from times p to T.
        for i in range(p, n, 1):
            x = y[cp.arange(i-1,i-p-1,-1),:].reshape(-1)
            # Define each y
            y[i, :] = A @ (beta.T @ x) + 0.1* err[i-p,:]
    #         print(y[i,:])
                    
    beta_true = beta @ mat_power(beta.T @ beta, -0.5)
    return y, p, d, beta_true


# %%
def one_run(n, ex, cov_str, pi=None, seed=None):
    y, p, d, beta_true = examples(n,ex,cov_str,pi,seed)
    s_true = np.where(norm(beta_true, axis=1, keepdims=True)>10e-5)[0]
    beta1 = ts_mddm(y,p,d)
    beta2, s_hat1 = smddm_l1(y,p,d)
    beta3, s_hatg = smddm(y,p,d)
    beta4 = ts_sir(y,p,d,H=10)
    beta5, _ = reduced_rank(y, p, d)

    ## general loss
    loss1 = norm(beta1 @ cp.linalg.inv(beta1.T @ beta1) @ beta1.T - beta_true @ cp.linalg.inv(beta_true.T @ beta_true) @ beta_true.T, 'fro')
    loss2 = norm(beta2 @ cp.linalg.inv(beta2.T @ beta2) @ beta2.T - beta_true @ cp.linalg.inv(beta_true.T @ beta_true) @ beta_true.T, 'fro')
    loss3 = norm(beta3 @ cp.linalg.inv(beta3.T @ beta3) @ beta3.T - beta_true @ cp.linalg.inv(beta_true.T @ beta_true) @ beta_true.T, 'fro')
    loss4 = norm(beta4 @ cp.linalg.inv(beta4.T @ beta4) @ beta4.T - beta_true @ cp.linalg.inv(beta_true.T @ beta_true) @ beta_true.T, 'fro')
    loss5 = norm(beta5 @ cp.linalg.inv(beta5.T @ beta5) @ beta5.T - beta_true @ cp.linalg.inv(beta_true.T @ beta_true) @ beta_true.T, 'fro')
    
    ## orthogonal
    beta1,_ = cp.linalg.qr(beta1)
    beta2,_ = cp.linalg.qr(beta2)
    beta3,_ = cp.linalg.qr(beta3)
    beta4,_ = cp.linalg.qr(beta4)
    beta5,_ = cp.linalg.qr(beta5)
    beta_true,_ = cp.linalg.qr(beta_true)

    ## vector correlation
    r1 = cp.sqrt(cp.linalg.det(beta1.T @ beta_true @ beta_true.T @ beta1))
    r2 = cp.sqrt(cp.linalg.det(beta2.T @ beta_true @ beta_true.T @ beta2))
    r3 = cp.sqrt(cp.linalg.det(beta3.T @ beta_true @ beta_true.T @ beta3))
    r4 = cp.sqrt(cp.linalg.det(beta4.T @ beta_true @ beta_true.T @ beta4))
    r5 = cp.sqrt(cp.linalg.det(beta5.T @ beta_true @ beta_true.T @ beta5))

    est_d = estimate_dim(y,p)
    est_l = estimate_lag(y, 30) if ex < 5 else estimate_lag(y, 10)
    
    fp1 = set(s_hat1.get())-set(s_true.get())
    fn1 = set(s_true.get())-set(s_hat1.get())  
    fpg = set(s_hatg.get())-set(s_true.get())
    fng = set(s_true.get())-set(s_hatg.get())   
    
    dict = {
        'size': n,
        'example':ex,
        'cov_str': cov_str,
        'loss_mddm': loss1.item(),
        'loss_lasso': loss2.item(),
        'loss_group': loss3.item(),
        'loss_sir': loss4.item(),
        'loss_rr': loss5.item(),
        'corr_mddm': r1.item(),
        'corr_lasso': r2.item(),
        'corr_group': r3.item(),
        'corr_sir': r4.item(),
        'corr_rr': r5.item(),
        'est_dim': est_d.item(),
        'est_lag': est_l.item(),
        'fp_lasso': len(fp1),
        'fn_lasso': len(fn1),
        'fp_group': len(fpg),
        'fn_group': len(fng),
        'lag': p,
        'seed': seed
    }
    df = pd.DataFrame(dict, index=[seed]).round(4)
    return df

# %%
start_time = time.time()
data_list = []
rep = 100
for n in [500,1000,1500, 2000,2500,3000]:
    for ex in range(3,5,1):
        for cov_str in range(1,3,1):
            print('Example:',ex,", n =",n,"cov_str =",cov_str)
            for seedid in range(rep):
                result = one_run(n,ex,cov_str,seed=seedid)
                data_list.append(result)

df = pd.concat([df for df in data_list], ignore_index=True)
end_time = time.time()
execution_time = end_time - start_time
print(f"simulation time run: {execution_time} seconds")

# %%
fname = f'simulation.pkl'
df.to_pickle(fname, compression='gzip')           # save dataframe as pkl