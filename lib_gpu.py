import cupy as cp
import numpy as np
from cupyx.scipy.sparse import linalg
from cupy.linalg import norm

###############################################################################

# %%
def mat_power(matrix, power):
    # Diagonalize the matrix
    eigenvalues, eigenvectors = cp.linalg.eigh(matrix)

    # Take the square root of the eigenvalues
    power_eigenvalues = (eigenvalues) ** power

    # Reconstruct the matrix from the square roots of the eigenvalues
    return cp.dot(eigenvectors, cp.dot(cp.diag(power_eigenvalues), cp.linalg.inv(eigenvectors)))

# # Example usage
# A = cp.array([[4, 1], [1, 4]])
# B = mat_power(A, -0.5)
# # print(B@B@A)

 
# %%
def wire(X, y):
    """Calculate sample kernel matrix for WIRE using CuPy for GPU acceleration.
    X: design matrix
    y: response vector
    """
    n = X.shape[0]
    y = y.reshape((n, -1))
    x0 = X - cp.mean(X, axis=0)
    q = y.shape[1]

    temp = cp.zeros((n, n, q))
    for i in range(q):
        temp[:, :, i] = cp.subtract(y[:, i].reshape(n, 1), y[:, i].reshape(1, n))
    D = cp.sqrt(cp.sum(cp.square(temp), 2))

    M = -(x0.T @ D @ x0) / ((n - 1) ** 2)
    return M

# %%
def ts_mddm(y, p, d=None):
    """
    Input:
        Y: n X q
        X: n X qp
        d: reduced rank
    Output:
        betaest: qp X d
    """
    # dimensions of y
    n, q = y.shape
    # dimension of x
    dim_x = p*q
    # construct x
    x = cp.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[cp.arange(i-1,i-p-1,-1),:].reshape(-1)
    # covariate
    sigma = cp.cov(x.T)
    # remove the first p values
    yt = y[p:,:]
    # MDDM
    M = wire(x, yt)
    
    if d is None:
        d = dim_x
    # Initial approximation
    Xi = cp.random.rand(dim_x, d)
    # Compute the eigenvalues and eigenvectors
    _, eigenvectors = linalg.lobpcg(M, Xi, B=sigma)
    # print("Eigenvalues:", eigenvalues)
    # print("Eigenvectors:\n", eigenvectors)

    beta1 = eigenvectors
    # print("est_beta:\n", beta1)

    # normalization
    if d == 1:
        beta1 = beta1/norm(beta1)
    else:
        beta1 = (beta1)@ mat_power(beta1.T @ beta1,-0.5)
    # print("normalised est_beta:\n", beta1)

    return beta1

# %%
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
    x = cp.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[cp.arange(i-1,i-p-1,-1),:].reshape(-1)
    # covariate
    sigma = cp.cov(x.T)
    # remove the first p values
    yt = y[p:,:]
    # MDDM
    Gam = wire(x, yt)
    lam_max = max(norm(Gam, axis = 1))
    if lam is None:
        lam = lam_max*0.3
    max_iter = 300
    t0 = norm(sigma, 2) ** (-2)
    t = 1
    M = cp.zeros((dim_x, dim_x))
    B = cp.zeros((dim_x, dim_x))

    # obj = np.zeros(max_iter)
    for i in range(max_iter):
        # soft-threshold update M
        B_old = B
        M_old = M
        temp = M - t0 * (sigma @ B @ sigma - Gam)
        w = cp.maximum(1 - lam * t0 / norm(temp, axis=1, keepdims=True), 0)
        w = w.reshape(-1)
        M = (w * temp.T).T
        # symmetrization
        M[:,w==0] = 0
        M = (M+M.T)/2

        # update step
        t_new = (1 + cp.sqrt(1 + 4 * t ** 2))/2

        # update B
        B = M + (t -1)/t_new * (M - M_old)

        # objective
        # obj[i] = 1/2 * cp.trace(M.T @ sigma @ M @ sigma) - cp.trace(M.T @ Gam) + lam * cp.sum(norm(M, axis=1, keepdims=True))

        if max(norm(B_old - B, 'fro'), norm(M_old - M, 'fro')) < 1e-8:
            break
    # check sparsity
    # print("w:", w)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = cp.linalg.eigh(M)
    # print("Eigenvalues:", eigenvalues)
    # print("Eigenvectors:\n", eigenvectors)

    # final estimate
    beta2 = cp.fliplr(eigenvectors)
    beta2 = beta2[:,:d]
    
    ind_nonzero = cp.where(w>10e-5)[0]
    return beta2, ind_nonzero

# %%
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
    x = cp.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[cp.arange(i-1,i-p-1,-1),:].reshape(-1)
    # covariate
    sigma = cp.cov(x.T)
    # remove the first p values
    yt = y[p:,:]
    # MDDM
    Gam = wire(x, yt)
    assert cp.all(cp.isclose(Gam - Gam.T,0))
    lam_max = cp.max(Gam)
    # print(lam_max)
    if lam is None:
        lam = lam_max*0.3

    max_iter = 300
    t0 = norm(sigma, 2) ** (-2)
    t = 1
    M = cp.zeros((dim_x, dim_x))
    B = cp.zeros((dim_x, dim_x))

    # obj = np.zeros(max_iter)
    for i in range(max_iter):
    #     print(i)
        # soft-threshold update M
        B_old = B
        M_old = M
        temp = M - t0 * (sigma @ B @ sigma - Gam)
        M = cp.maximum(temp - lam * t0, 0) * cp.sign(temp)
        assert cp.all(cp.isclose(M - M.T,0))
    #     M = (M+M.T)/2

        # update step
        t_new = (1 + cp.sqrt(1 + 4 * t ** 2))/2

        # update B
        B = M + (t -1)/t_new * (M - M_old)

        # objective
        # obj[i] = 1/2 * cp.trace(M.T @ sigma @ M @ sigma) - cp.trace(M.T @ Gam) + lam * cp.sum(norm(M, axis=1, keepdims=True))

        if max(norm(B_old - B, 'fro'), norm(M_old - M, 'fro')) < 1e-8:
            break
    # check sparsity
    # print("w:", w)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = cp.linalg.eigh(M)
    # print("Eigenvalues:", eigenvalues)
    # print("Eigenvectors:\n", eigenvectors)

    # final estimate
    beta2 = cp.fliplr(eigenvectors)
    beta2 = beta2[:,:d]

    ind_nonzero = cp.where(norm(beta2, axis=1, keepdims=True)>10e-5)[0]
    return beta2, ind_nonzero

# %%
def estimate_lag(y, lag_max = None):
    n, q = y.shape
#     lag_max = 20
    if lag_max is None:
        lag_max = min(int(n/2),50)

    # Candidate lag sequence
    pseq = cp.arange(1, lag_max+1, 1)

    # Empty criteria vector
    cri = cp.zeros(lag_max)

    # Perform WIRE 
    for j in range(lag_max):
        pp = int(pseq[j])
#         print("check lag:",pp)
        # dimension of x
        dim_x = pp * q
        # construct x
        x = cp.empty((n-pp, dim_x)) #cp.zeros((n-pp, dim_x))
        for i in range(pp, n, 1):
            x[i-pp,:] = y[cp.arange(i-1,i-pp-1,-1),:].reshape(-1)
        # covariate
        sigma = cp.cov(x.T)
        # remove the first p values
        yt = y[pp:,:]
        Mt = wire(x, yt)

        # Initial approximation
        Xi = cp.random.rand(dim_x, dim_x)
        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = linalg.lobpcg(Mt, Xi, B=sigma)

        beta1 = eigenvectors
        betaest = beta1
        cri[j] = - cp.trace(betaest.T @ Mt @ betaest)

    rel_ratio = cp.abs(cri[1:] - cri[:len(cri)-1]) / cp.abs(cri[1:])
    ind = cp.where(rel_ratio > 0.05)[0]
    est_p = pseq[ind[len(ind)-1] + 1]
#     print('Estimated lag:', est_p)

    return est_p

# %%
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
	
    Return: A resampled version, or "replicate" of data
    """
    num_obs = cp.shape(_data)[0]
    num_dims = cp.ndim(_data)
    assert num_dims == 1 or num_dims == 2, "Input data must be a 1 or 2D array"

    if num_dims == 1:
        wrapped_data = cp.concatenate((_data, _data))
    elif num_dims == 2:
        wrapped_data = cp.vstack((_data, _data))

    assert p > 0 and p <= 1, "p must be in (0,1]"

    if seed is not None:
        cp.random.seed(seed=seed)

    choices = cp.random.randint(0, num_obs, num_obs)
    unifs = cp.random.uniform(0, 1, num_obs)

    indices = -cp.ones(num_obs, dtype=int)
    indices[0] = 0

    for i in cp.arange(1, num_obs):
        if unifs[i] > p:
            indices[i] = indices[i - 1] + 1
        else:
            indices[i] = choices[i]

    if num_dims == 1:
        resampled_data = wrapped_data[indices]
    elif num_dims == 2:
        resampled_data = wrapped_data[indices, :]

    return resampled_data

# resample(y[:10,:], 0.02)

# %%
def estimate_dim(y, p):
    # dimensions of y
    n, q = y.shape
    # dimension of x
    dim_x = p*q
    # construct x
    x = cp.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[cp.arange(i-1,i-p-1,-1),:].reshape(-1)
    # covariate
    sigma = cp.cov(x.T)
    # remove the first p values
    yt = y[p:,:]
    # MDDM
    M = wire(x, yt)

    k = min(5, dim_x)
    # Initial approximation
    Xi = cp.random.rand(dim_x, k)
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.lobpcg(M, Xi, B=sigma)
    beta_hat = eigenvectors
    beta_hat = beta_hat @ mat_power(beta_hat.T @ beta_hat, -1/2)
    value = eigenvalues

    # Resample and decompose bootstrap kernel matrix
    fk = cp.zeros(k)
    nbs = 100

    for j in range(nbs):
        # Resample for time series (resample needs to be CuPy-compatible)
        y_star = resample(y, 0.02)
        beta_b = ts_mddm(y_star,p,k)

        # Calculate f0
        for i in range(k-1):
            fk[i+1] += (1 - cp.abs(cp.linalg.det(beta_hat[:, 0:(i+1)].T @ beta_b[:, 0:(i+1)]))) / nbs

    # The final criteria
    eta = 0.3
    gn = eta * fk / (cp.sum(fk)) + (1-eta)*value / (cp.sum(value))
    est_d = cp.argmin(gn)
#     print("Estimated dimension", est_d)
    return est_d

# %%
def reduced_rank(y, p, d):
    """
    Input: 
        y: q-dimensional series
        p: lags
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
    x = cp.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[cp.arange(i-1,i-p-1,-1),:].reshape(-1)
    # remove the first p values
    yt = y[p:,:]

    SigEps = 1/n * (yt.T @ yt - yt.T @ x @ cp.linalg.inv(x.T @ x) @ x.T @ yt)
    GamXY = cp.cov(x.T, yt.T)
    GamX = GamXY[:dim_x, :dim_x]
    Gams = GamXY[:dim_x, dim_x:]

    SigHalf = mat_power(SigEps, 0.5)  # This needs to be replaced or adapted for CuPy
    GamXinv = cp.linalg.inv(GamX)
    AA = SigHalf @ Gams.T @ GamXinv @ Gams @ SigHalf
    VV = cp.linalg.eigh(AA)[1]
    V = VV[:, ::-1]
    V = V[:, :d]
    B_est = V.T @ SigHalf @ Gams.T @ GamXinv
    SigInvHalf = mat_power(SigEps, -0.5)  # This needs to be replaced or adapted for CuPy
    A_est = SigInvHalf @ V
    
    return B_est.T, A_est

# %%
def sir(x, y, H=5):
    n, p = x.shape
    M = cp.zeros((p, p))
    X0 = x - cp.mean(x, axis=0)
    YI = cp.argsort(y.reshape(-1))
    
    # Handling array splitting and looping in CuPy
    # Note: cp.array_split behaves similarly to np.array_split
    A = cp.array_split(YI, H)  # split into H parts

    for i in range(H):
        tt = X0[A[i]].reshape(-1, p)
        Xh = cp.mean(tt, axis=0).reshape(p, 1)
        M = M + Xh @ Xh.T * len(A[i]) / n
    return M

def ts_sir(y,p,d,H=5):
    """
    Input: 
        Y: n X q
        X: n X qp
        d: reduced rank
    Output:
        betaest: qp X d
    """
    # dimensions of y
    n, q = y.shape
    # dimension of x
    dim_x = p*q
    # construct x
    x = cp.zeros((n-p, dim_x))
    for i in range(p, n, 1):
        x[i-p,:] = y[cp.arange(i-1,i-p-1,-1),:].reshape(-1)
    # remove the first p values
    yt = y[p:,:]
    
    if yt.shape[1] > 1:
        M = cp.zeros((dim_x, dim_x))
        for i in range(yt.shape[1]):
            # Ensure that sir function is compatible with CuPy
            M += sir(x, yt[:, i], H)
    else:
        # Ensure that sir function is compatible with CuPy
        M = sir(x, yt, H)
    
    sigma = cp.cov(x.T)
    
    # Ensure eigh function works with CuPy arrays
    # Initial approximation
    Xi = cp.random.rand(dim_x, d)
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = linalg.lobpcg(M, Xi, B=sigma)
    beta1 = eigenvectors
    
    # normalization
    if d == 1:
        beta1 = beta1/norm(beta1)
    else:
        beta1 = (beta1)@ mat_power(beta1.T @ beta1,-0.5)
    return beta1