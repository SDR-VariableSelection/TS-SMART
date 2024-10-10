import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.linalg import norm
from scipy.linalg import eigh
from scipy.linalg import svd
from scipy.linalg import sqrtm
from scipy import stats
from sklearn.model_selection import KFold # import KFold
from sklearn.utils import resample
import random
from joblib import Parallel, delayed
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from scipy.optimize import linprog
import statsmodels.api as sm

def wire(X, y, metric="Euclidean"):   
    """calculate sample kernel matrix for WIRE
    X: design matrix
    y: response vector
    metric: user-defined metric
    """ 
    n = X.shape[0]
    y = y.reshape((n,-1))
    x0 = X - np.mean(X, axis=0)            
    q = y.shape[1]
    if metric == "Euclidean":
        temp = np.zeros((n, n, q))
        for i in range(q):
            temp[:,:,i] = np.subtract(y[:,i].reshape(n,1), y[:,i].reshape(1,n))
        D = np.sqrt(np.sum(np.square(temp),2))
    if metric == "Wasserstein":
        temp = np.zeros((n, n, q))
        for i in range(q):
            temp[:,:,i] = np.subtract(y[:,i].reshape(n,1), y[:,i].reshape(1,n))
        D = np.sqrt(np.mean(np.square(temp),2))
#         indnon0 = np.nonzero(D)
#         D = D/(1+D)
    if metric == "Geodesic":
        D = np.arccos(np.minimum(y@y.T, 1)) #np.arccos(y @ y.T)
    M = -(x0.T @ D @ x0)/((n-1)**2)
    return M, D

def loss(A, A_hat):
    '''assessment of estimates
    A: true parameter
    A_hat: estimate
    '''
    loss_general = norm(A_hat @ la.inv(A_hat.T @ A_hat) @ A_hat.T - A @ la.inv(A.T @ A) @ A.T)
    return loss_general


from collections import defaultdict
# https://gist.github.com/josef-pkt/2279868
def resample(_data, p, seed=None):
    """
    Performs a stationary block bootstrap resampling of elements from a time 
    series, or rows of an input matrix representing a multivariate time series
	
    Inputs:
        data - An MxN numerical array of data to be resampled. It is assumed
               that rows correspond to measurement observations and columns to 
	       items being measured. 1D arrays are allowed.
        p    - A scalar float in (0,1] defining the parameter for the geometric
               distribution used to draw block lengths. The expected block 
	       length is then 1/p				
    Keywords:
        seed - A scalar integer defining the seed to use for the underlying
	   random number generator.
	
    Return:
        A three element list containing
	- A resampled version, or "replicate" of data
	- A length M array of integers serving as indices into the rows
	  of data that were resampled, where M is the number of rows in 
	  data. Thus, if indices[i] = k then row i of the replicate data 
	  contains row k of the original data.
	- A dictionary containing a mapping between indices into data and
	  indices into the replicate data. Specifically, the keys are the
	  indices for unique numbers in data and the associated dict values 
	  are the indices into the replicate data where the numbers are.
    Example:			
        In [1]: import numpy as np
        In [2]: x = np.random.randint(0,20, 10)
        In [3]: import stationary_block_bootstrap as sbb
        In [4]: x_star, x_indices, x_indice_dict = sbb.resample(x, 0.333333)
        In [5]: x
        Out[5]: array([19,  2,  9,  9,  9, 10,  2,  2,  0, 11])
        In [6]: x_star
        Out[6]: array([19, 11,  2,  0, 11, 19,  2,  2, 19,  2])
        In [7]: x_indices
        Out[7]: array([0, 9, 7, 8, 9, 0, 6, 7, 0, 1])
        So, x_indices[1] = 9 means that the 1th element of x_star corresponds 
        to the 9th element of x
		
        In [8]: x_indice_dict
        Out[8]: {0: [0, 5, 8], 7: [2, 6, 7, 9], 8: [3], 9: [1, 4]}
	
        So, x[0] = 19 occurs in position 0, 5, and 8 in x_star. Likewise, 
        x[9] = 11 occurs in positions 1 and 4 in x_star
	
"""
    num_obs = np.shape(_data)[0]
    num_dims = np.ndim(_data)
    assert num_dims == 1 or num_dims == 2, "Input data must be a 1 or 2D array"
    #There is a probably smarter way to wrap the series without doubling
    #the data in memory; the approach below is easy but wasteful
    if num_dims == 1:
        wrapped_data = np.concatenate((_data, _data)) 
    elif num_dims == 2:
        wrapped_data = np.row_stack((_data, _data)) 
	
    assert p > 0 and p <=1, "p must be in (0,1]"
	
    if seed is not None:
        np.random.seed(seed=seed)

    #Make the random variables used in the resampling ahead of time. Could be
    #problematic for memory if num_obs is huge, but doing it this way cuts down 
    #on the function calls in the for loop below...
    choices = np.random.randint(0, num_obs, num_obs)
    unifs = np.random.uniform(0, 1, num_obs)
	
    #Let x and x* be length-n vectors with x*[0] = x[0]. 
    #Then x*[1] = x[1] with probability 1-p. With probability p, x*[1] will
    #equal a random i for x[i]. The expected run length is 1/p = "block length"
    indices = -np.ones(num_obs, dtype=int)
    indices[0] = 0
		
    for i in np.arange(1, num_obs):
        if (unifs[i] > p): 
            indices[i] = indices[i-1] + 1 
        else:
            indices[i] = choices[i]

    if num_dims == 1:		
        resampled_data = wrapped_data[indices]   
        index_to_data_map = dict((x, i) for i, x in enumerate(wrapped_data))
        bootstrap_indices = map(index_to_data_map.get, resampled_data)
    elif num_dims == 2:
        #Mapping is the same for each column with respect to which rows are
        #resampled, so just consider one variable when mapping indices to data...
        resampled_data = wrapped_data[indices, :]   
        index_to_data_map = dict((x, i) for i, x in enumerate(wrapped_data[:,0]))
        bootstrap_indices = map(index_to_data_map.get, resampled_data[:,0])
		
    bootstrap_indices = [index % num_obs for index in bootstrap_indices]
	
    #The code below is making a dictionary mapping of observations resampled to
    #where in the array of indices that observation shows up. Some observations
    #are resampled multiple times, others not at all, in any given replicate
    #data set. The straight-forward code is
    # try:
    #   items = dict[key]
    # except KeyError:
    #   dict[key] = items = [ ]
    #   items.append(value)
    
    index_occurences = defaultdict(list)
    for pos, index in enumerate(bootstrap_indices):
        index_occurences[index].append(pos)
	
    index_dict = dict(index_occurences)

    #Need to make the indices we save be bounded by 0:num_obs. For example, 
    #data[0,:] = data[num_obs,:]  and data[1,:] = data[num_obs+1,*] etc     
    #because we wrapped the data. But, with respect to the data arrays used 
    #elsewhere, an index of num_obs+1 is out of bounds, so num_obs should be 
    #converted to 0, num_obs+1 to 1, etc...   

    return [resampled_data, indices % num_obs, index_dict] 
    #end resample()
    

def reduced_rank(y, p, d):
    """
    Input: 
        Y: n X q
        d: reduced rank
    Output:
        B_est.T: qp X d
        A_est: q X d
    """
    # dimensions of y
    n, q = y.shape
    # dimension of x
    dim_x = p*q
    # construct x
    x = np.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[np.arange(i-1,i-p-1,-1),:].reshape(-1)
    # remove the first p values
    yt = y[p:,:]

    SigEps = 1/n * ( yt.T @ yt - yt.T @ x @ la.inv(x.T @ x) @ x.T @ yt)
    GamXY = np.cov(x.T, yt.T)
    GamX = GamXY[:dim_x, :dim_x]
    Gams = GamXY[:dim_x, dim_x:]
    SigHalf = fractional_matrix_power(SigEps, 0.5)
    GamXinv = la.inv(GamX)
    AA = SigHalf @ Gams.T @ GamXinv @ Gams @ SigHalf
    VV = eigh(AA)[1]
    V = VV[:,::-1]
    V = V[:,:d]
    B_est = V.T @ SigHalf @ Gams.T @ GamXinv
    SigInvHalf = fractional_matrix_power(SigEps, -0.5)
    A_est = SigInvHalf @ V

    return B_est.T, A_est

def time_series_reduced_rank_regression(Y, rank, lags):
    """
    Perform time-series reduced rank regression between predictors X and responses Y.
    
    Parameters:
    X (numpy.ndarray): Predictor time-series matrix.
    Y (numpy.ndarray): Response time-series matrix.
    rank (int): The rank to which the regression matrix is reduced.
    lags (int): Number of lagged terms to consider in the model.
    
    Returns:
    numpy.ndarray: The estimated coefficients matrix for each lag.
    """
    # Create lagged versions of X
    X_lagged = sm.tsa.lagmat(Y, maxlag=lags, trim='both')
    Y = Y[lags:]  # Adjust Y to match the shape of X_lagged
    
    # SVD of X_lagged
    U, s, Vt = np.linalg.svd(X_lagged, full_matrices=False)
    S_inv = np.diag(1 / s[:rank])  # Only take 'rank' number of singular values
    
    # Reduced rank estimation
    Vt_r = Vt[:rank, :]
    B = Vt_r.T @ S_inv @ U.T[:rank, :] @ Y
    return B

def ts_mddm(y,p,d=None):
    """
    Input:
        Y: n X q
        p: lags
        d: reduced rank
    Output:
        betaest: qp X d
    """
    # dimensions of y
    n, q = y.shape
    # dimension of x
    dim_x = p*q
    # construct x
    x = np.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[np.arange(i-1,i-p-1,-1),:].reshape(-1)
    # covariate
    sigma = np.cov(x.T)
    # remove the first p values
    yt = y[p:,:]
    # MDDM
    M = wire(x, yt)[0]
    if d is None:
        d = dim_x

    vector = eigh(M, sigma)[1]
    eta = np.fliplr(vector)
    betaest = eta[:,:d]
    return betaest

def smddm(y, p, d, lam=None):
    """
    Input:
        Y: n X q
        p: lags
        d: reduced rank
    Output:
        beta: qp X d
    """
    # dimensions of y
    n, q = y.shape
    # dimension of x
    dim_x = p*q
    # construct x
    x = np.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[np.arange(i-1,i-p-1,-1),:].reshape(-1)
    # covariate
    sigma = np.cov(x.T)
    # remove the first p values
    yt = y[p:,:]
    # MDDM
    Gam = wire(x, yt)[0]
    lam_max = max(norm(Gam, axis = 1))
    if lam is None:
        lam = lam_max*0.3
    max_iter = 300
    t0 = norm(sigma, 2) ** (-2)
    t = 1
    M = np.zeros((dim_x, dim_x))
    B = np.zeros((dim_x, dim_x))

    obj = np.zeros(max_iter)
    for i in range(max_iter):
        # soft-threshold update M
        B_old = B
        M_old = M
        temp = M - t0 * (sigma @ B @ sigma - Gam)
        w = np.maximum(1 - lam * t0 / norm(temp, axis=1), 0)
        w = w.reshape(-1)
        M = (w * temp.T).T
        # symmetrization
        M[:,w==0] = 0
        M = (M+M.T)/2

        # update step
        t_new = (1 + np.sqrt(1 + 4 * t ** 2))/2

        # update B
        B = M + (t -1)/t_new * (M - M_old)

        # objective
        obj[i] = 1/2 * np.trace(M.T @ sigma @ M @ sigma) - np.trace(M.T @ Gam) + lam * np.sum(norm(M, axis=1))
        t = t_new

        if max(norm(B_old - B, 'fro'), norm(M_old - M, 'fro')) < 1e-8:
            break
    # check sparsity
    # print("w:", w)

    # Compute the eigenvalues and eigenvectors
    eigenvectors = eigh(M)[1]
    # print("Eigenvectors:\n", eigenvectors)

    # final estimate
    beta2 = np.fliplr(eigenvectors)
    beta2 = beta2[:,:d]
    
    ind_nonzero = np.where(w>10e-5)[0]
    return beta2, ind_nonzero

def smddm_l1(y, p, d, lam=None):
    """
    Input:
        Y: n X q
        p: lags
        d: reduced rank
    Output:
        beta: qp X d
    """
    # dimensions of y
    n, q = y.shape
    # dimension of x
    dim_x = p*q
    # construct x
    x = np.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[np.arange(i-1,i-p-1,-1),:].reshape(-1)
    # covariate
    sigma = np.cov(x.T)
    # remove the first p values
    yt = y[p:,:]
    # MDDM
    Gam = wire(x, yt)[0]
    assert np.all(np.isclose(Gam - Gam.T,0))
    lam_max = np.max(Gam)
    # print(lam_max)
    if lam is None:
        lam = lam_max*0.3

    max_iter = 300
    t0 = norm(sigma, 2) ** (-2)
    t = 1
    M = np.zeros((dim_x, dim_x))
    B = np.zeros((dim_x, dim_x))

    obj = np.zeros(max_iter)
    for i in range(max_iter):
    #     print(i)
        # soft-threshold update M
        B_old = B
        M_old = M
        temp = M - t0 * (sigma @ B @ sigma - Gam)
        M = np.maximum(temp - lam * t0, 0) * np.sign(temp)
        assert np.all(np.isclose(M - M.T,0))
    #     M = (M+M.T)/2

        # update step
        t_new = (1 + np.sqrt(1 + 4 * t ** 2))/2

        # update B
        B = M + (t -1)/t_new * (M - M_old)

        # objective
        obj[i] = 1/2 * np.trace(M.T @ sigma @ M @ sigma) - np.trace(M.T @ Gam) + lam * np.sum(np.abs(M))

        if max(norm(B_old - B, 'fro'), norm(M_old - M, 'fro')) < 1e-8:
            break
    # check sparsity
    # print("w:", w)

    # Compute the eigenvalues and eigenvectors
    eigenvectors = eigh(M)[1]
    # print("Eigenvalues:", eigenvalues)
    # print("Eigenvectors:\n", eigenvectors)

    # final estimate
    beta2 = np.fliplr(eigenvectors)
    beta2 = beta2[:,:d]

    ind_nonzero = np.where(norm(beta2, axis=1)>10e-5)[0]
    return beta2, ind_nonzero


def sir(X, y, H=5):
    n, p = X.shape
    M = np.zeros((p, p))
    X0 = X - np.mean(X, axis = 0)
    YI = np.argsort(y.reshape(-1))
    A = np.array_split(YI, H) # split into 5 parts
    for i in range(H):
        tt = X0[A[i],].reshape(-1,p)
        Xh = np.mean(tt, axis = 0).reshape(p, 1)
        M = M + Xh @ Xh.T * len(A[i])/n
    return M

def ts_sir(y, p, d, H = 5):
    """
    Input: 
        Y: n X q
        X: n X qp
        d: reduced rank
    Output:
        betaest: qp X d
    """
    n, q = y.shape
    # dimension of x
    dim_x = p*q
    # construct x
    x = np.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[np.arange(i-1,i-p-1,-1),:].reshape(-1)
    # covariate
    sigma = np.cov(x.T)
    # remove the first p values
    yt = y[p:,:]

    if yt.shape[1]>1:
        M = np.zeros((dim_x,dim_x))
        for i in range(yt.shape[1]):
            M += sir(x, yt[:,i],H)
    else:
        M = sir(x,yt,H)
    sigma = np.cov(x.T)
    vector = eigh(M, sigma)[1]
    eta = np.fliplr(vector)
    betaest = eta[:,:d]
    return betaest


def estimate_lag(y, lag_max=10):
    n, q = y.shape

    if lag_max is None:
        lag_max = min(int(n/2),50)

    # candidate lag candidate sequence
    pseq = np.arange(1, lag_max+1, 1)
    # emtpy criteria vector 
    cri = np.zeros(len(pseq))
    
    # perform WIRE
    for pt in pseq:
        Xt = np.zeros((n-pt, pt*q))
        Yt = y[pt:n,:]
        for i in np.arange(pt, n, 1):
            Xt[i-pt,:] = y[np.arange(i-1,i-pt-1,-1),:].reshape(-1)

        Mt = wire(Xt,Yt, "Euclidean")[0]
        sigma = np.cov(Xt.T)
        vector = eigh(Mt, sigma)[1]
        eta = np.fliplr(vector)
        # use estimated eta, which is qp times dt. 
        betaest = eta#[:,:dt]
        cri[pt-1] = - np.trace(betaest.T @ Mt @ betaest)

    plt.plot(pseq, cri)
    rel_ratio = np.abs(cri[1:]-cri[:(len(cri)-1)])/np.abs(cri[1:])
    ind = np.where(rel_ratio>0.05)[0]
    est_p = pseq[ind[len(ind)-1]+1]
    print('estimated p:', est_p)

    return est_p

def estimate_dim(y, p, metric="Euclidean", ker=wire):
#     metric = "Euclidean"
    # decomposite kernel matrix to obtain eigenvalues and vectors
    n,q = y.shape
    Yt = y[p:n,:]
    Xt = np.zeros((n-p, p*q))
    for i in np.arange(p, n, 1):
        Xt[i-p,:] = y[np.arange(i-1,i-p-1,-1),:].reshape(-1)

    M = ker(Xt,Yt,metric)[0]
    n,pq = Xt.shape
    sigma = np.cov(Xt.T)
    k = min(10, pq)
    value, beta_hat = eigh(M, sigma, subset_by_index=(pq-k, pq-1))
    beta_hat = np.fliplr(beta_hat)
    beta_hat = beta_hat @ sqrtm(la.inv(beta_hat.T @ beta_hat))
    value = value[::-1]
    
    # resample and decomposite bootstrap kernel matrix
    fk = np.zeros(k)
    nbs = 100
    xy = np.concatenate((Yt,Xt),axis = 1)

    for j in range(nbs):    
#       # resample for time series
        xy_star, _, _ = resample(xy, 0.02)
        # new sample
        y_bs = xy_star[:,:q]
        X_bs = xy_star[:,q:]
        # obtain eigenvalues and vectors
        sigma_bs = np.cov(X_bs.T)
        M_bs,_ = ker(X_bs, y_bs, metric)
        _, beta_b = eigh(M_bs, sigma_bs, subset_by_index=(pq-k, pq-1))
        beta_b = np.fliplr(beta_b)
        beta_b = beta_b @ sqrtm(la.inv(beta_b.T @ beta_b))

        # calculate f0
        for i in range(k-1):
            fk[i+1] += (1-np.abs(la.det(beta_hat[:,0:(i+1)].T @ beta_b[:,0:(i+1)])))/nbs
    # the final criteria
    gn = fk/(1+np.sum(fk)) + value/(1+np.sum(value))
    d = np.argmin(gn)
    return d, gn

def VAR_highdim(y, p, lam = 0.001):
    n, q = y.shape
    # dimension of x
    dim_x = p*q
    # construct x
    x = np.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[np.arange(i-1,i-p-1,-1),:].reshape(-1)

    S = (x.T @ x)/x.shape[0]
    S1 = (x[:-1,:].T @ x[1:,:])/(x.shape[0]-1)

    # lam = 0.001
    beta_hat = np.zeros((p*q, q))
    # Coefficients for the inequality constraints Ax <= b
    A = np.zeros((2*p*q, 2*p*q))
    A[:p*q, :p*q] = S
    A[p*q:, :p*q] = -S
    A[:p*q, p*q:] = -S
    A[p*q:, p*q:] = S
    c = np.ones(2*p*q)
    
    def solve_lp(j):
        b = np.zeros(2 * p * q)
        b[:p * q] = S1[:, j] + lam
        b[p * q:] = -S1[:, j] + lam

        # Solving the linear programming problem
        result = linprog(c, A_ub=A, b_ub=b, method='simplex')
        return result.x
    
    from joblib import Parallel, delayed
    result = Parallel(n_jobs=-1, verbose=5)(
        delayed(solve_lp)(j) for j in range(q)
        )

    for j in range(q):
        x = result[j]
        for i in range(p * q):
            beta_hat[i, j] = x[i] - x[i + p * q]

    return beta_hat

# def VAR_highdim(y, p, lam = 0.001):
#     n, q = y.shape
#     # dimension of x
#     dim_x = p*q
#     # construct x
#     x = np.zeros((n-p, dim_x))
#     for i in range(p, n, 1):
#         x[i-p,:] = y[np.arange(i-1,i-p-1,-1),:].reshape(-1)

#     S = (x.T @ x)/x.shape[0]
#     S1 = (x[:-1,:].T @ x[1:,:])/(x.shape[0]-1)

#     # lam = 0.001
#     beta_hat = np.zeros((p*q, q))
#     # Coefficients for the inequality constraints Ax <= b
#     A = np.zeros((2*p*q, 2*p*q))
#     A[:p*q, :p*q] = S
#     A[p*q:, :p*q] = -S
#     A[:p*q, p*q:] = -S
#     A[p*q:, p*q:] = S
#     c = np.ones(2*p*q)
#     for j in range(q):
#         b = np.zeros(2*p*q)
#         b[:p*q] = S1[:,j] + lam
#         b[p*q:] = -S1[:,j] + lam
#         # Solving the linear programming problem
#         result = linprog(c, A_ub=A, b_ub=b, method='simplex')

#         # # Displaying the results
#         # print("Optimal values of x and y:", result.x)
#         # print("Maximum value of the objective function:", -result.fun)  # Negating the result for maximization
#         # print("Status of the optimization:", result.message)
        
#         for i in range(p*q):
#             beta_hat[i, j] = result.x[i] - result.x[i+p*q]
#     return beta_hat

def fit_model(Xtr, Ytr, Xtest, bb_model_index):
    ## Choose a black-box machine learning model (0,1,2,3,4,5)
    if bb_model_index == 0:
        # linear regression
        black_box = LinearRegression()
    elif bb_model_index == 1:
        # kernel ridge regression
        black_box = KernelRidge(kernel='rbf', gamma=0.1)
    elif bb_model_index==2:
        # Random forest
        black_box = RandomForestRegressor(n_estimators=100, min_samples_split=10, random_state=2023)
    elif bb_model_index==3:
        # Random forest with more aggressive splits
        black_box = RandomForestRegressor(n_estimators=100, min_samples_split=1, random_state=2023)
    elif bb_model_index==4:
        # Support vector machine
        black_box = SVR(kernel='rbf', degree=3)
    elif bb_model_index==5:
        # Standard scikit-learn neural network
        black_box = MLPRegressor(hidden_layer_sizes=(200,), max_iter=1000, random_state=2023)
    else:
        print("Error: unknown machine learning model")
        black_box = None
    # Fit a random forest to the data
    black_box.fit(Xtr, Ytr)
    y_hat = black_box.predict(Xtest)
    # mse_bb = norm(Ytest-f_hat, 'fro')
    return y_hat