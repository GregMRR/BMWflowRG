"""Regulators to plug into the flow.

Each regulator `name(x, ...)` comes with its derivative `name_(x, ...)`.
`x` is the momentum q, not q^2; unless stated otherwise the derivative is taken with respect to y = q^2.
Wrap the extra arguments in a lambda: `r = lambda x: wett_alpha_beta(x, 2, 1)`."""
import numpy as np

#x is momentum, not momentum^2

def float32francesco(x): #15->7 AND 0.001->0.02
    """Regulator 2/(exp(q^2) - 1), set to 1/q^2 for q <= 0.02 and to 0 for q >= 7 (float32-safe cuts)."""
    R = np.zeros(np.shape(x))
    R[x<=0.02] = 1 / x[x<=0.02]**2
    R[np.logical_and(x>0.02, x < 7)] = 2 / (np.exp(x[np.logical_and(x>0.02, x < 7)]**2) - 1)
    return R

def float32francesco_(x):
    """Derivative of `float32francesco` with respect to q^2 on 0.02 < q < 7; 0 for q >= 7.

    For q <= 0.02 it returns -2/q^2 (the source comment notes that q^4 is intended)."""
    R_ = np.zeros(np.shape(x))
    R_[x<=0.02] = -2 / x[x<=0.02]**2 # it must be x**4
    R_[np.logical_and(x>0.02, x < 7)] = - 2 / (2 * (np.cosh(x[np.logical_and(x>0.02, x < 7)]**2) - 1))
    R_[x>=7] =  0.
    return R_


def francesco(x):
    """Regulator 2/(exp(q^2) - 1), set to 1/q^2 for q <= 0.001 and to 0 for q >= 15."""
    R = np.zeros(np.shape(x))
    R[x<=0.001] = 1 / x[x<=0.001]**2
    R[np.logical_and(x>0.001, x < 15)] = 2 / (np.exp(x[np.logical_and(x>0.001, x < 15)]**2) - 1)
    return R

def francesco_(x):
    """Derivative of `francesco` with respect to q^2 on 0.001 < q < 15; 0 for q >= 15.

    For q <= 0.001 it returns -2/q^2 (the source comment notes that q^4 is intended; see `francescoCorr_`)."""
    R_ = np.zeros(np.shape(x))
    R_[x<=0.001] = -2 / x[x<=0.001]**2 # it must be x**4
    R_[np.logical_and(x>0.001, x < 15)] = - 2 / (2 * (np.cosh(x[np.logical_and(x>0.001, x < 15)]**2) - 1))
    R_[x>=15] =  0.
    return R_

def francescoCorr_(x):
    """Derivative of `francesco` as `francesco_`, with -2/q^4 for q <= 0.001."""
    R_ = np.zeros(np.shape(x))
    R_[x<=0.001] = -2 / x[x<=0.001]**4 # !!!
    R_[np.logical_and(x>0.001, x < 15)] = - 2 / (2 * (np.cosh(x[np.logical_and(x>0.001, x < 15)]**2) - 1))
    R_[x>=15] =  0.
    return R_


def wett(x):
    """Wetterich regulator 1/(exp(q^2) - 1), set to 0 for q >= 15 to avoid an overflow."""
    R = np.zeros(np.shape(x))
    R[x < 15] = 1 / (np.exp(x[x < 15]**2) - 1)  #to avoid overflow in exp for big x
    R[x>=15] = 0.
    return R

def wett_(x):
    """Derivative of `wett` with respect to q^2: -exp(q^2)/(exp(q^2) - 1)^2; 0 for q >= 15."""
    R_ = np.zeros(np.shape(x))
    #R_[x<=0.001] = -2 / x[x<=0.001]**2
    R_[x < 15] = - np.exp(x[x < 15]**2) / ((np.exp(x[x < 15]**2) - 1)**2) 
    R_[x>=15] = 0.
    return R_


def wett_alpha(x, alpha):
    """Regulator alpha/(exp(q^2) - 1), set to 0 for q >= 16."""
    R = np.zeros(np.shape(x))
    R[x < 16] = alpha / (np.exp(x[x < 16]**2) - 1)
    R[x>=16] = 0.
    return R

def wett_alpha_(x, alpha):
    """Derivative of `wett_alpha` with respect to q^2: -alpha exp(q^2)/(exp(q^2) - 1)^2; 0 for q >= 15."""
    R_ = np.zeros(np.shape(x))
    #R_[x<=0.001] = -2 / x[x<=0.001]**2
    R_[x < 15] = - alpha * np.exp(x[x < 15]**2) / ((np.exp(x[x < 15]**2) - 1)**2) 
    R_[x>=15] = 0.
    return R_
    
    
def wett_alpha_beta(x, alpha, beta):
    """Regulator alpha/(exp(beta q^2) - 1), set to 0 for q >= 15. The examples use alpha = 2, beta = 1."""
    R = np.zeros(np.shape(x))
    R[x<15] = alpha / (np.exp(beta*x[x < 15]**2) - 1)
    R[x>=15] = 0.
    return R
    
def wett_alpha_beta_(x, alpha, beta):
    """Derivative of `wett_alpha_beta` with respect to q^2: -alpha beta exp(beta q^2)/(exp(beta q^2) - 1)^2; 0 for q >= 15."""
    R_ = np.zeros(np.shape(x))
    R_[x<15] = - alpha * beta * np.exp(beta*x[x < 15]**2) / ((np.exp(beta*x[x < 15]**2) - 1)**2) 
    R_[x>=15] = 0.
    return R_
    
    
def exp_alpha_beta(x, alpha, beta):
    """Regulator alpha exp(-beta q^2)."""
    R = alpha*np.exp(-beta*x**2)
    return R
    
def exp_alpha_beta_(x, alpha, beta):
    """Derivative of `exp_alpha_beta` with respect to q^2: -alpha beta exp(-beta q^2)."""
    R_ = -alpha*beta*np.exp(-beta*x**2)
    return R_


def exp(x): #for test purposes
    """Test regulator exp(-q) (a function of q, not q^2), set to 0 for q >= 2."""
    R = np.zeros(np.shape(x))
    R[x < 2]= np.exp(-x[x < 2])
    R[x>=2] = 0.
    return R

def exp_(x):
    """Derivative of `exp` with respect to q: -exp(-q); 0 for q >= 2."""
    R = np.zeros(np.shape(x))
    R[x < 2] = - np.exp(-x[x < 2])
    R[x>=2] = 0.
    return R


def exp1(x, alpha, beta): 
    """Regulator alpha exp(-beta q^2)/q^2."""
    R = alpha * np.exp(-beta * x**2) / x**2
    return R

def exp1_(x, alpha, beta):
    """Derivative of `exp1` with respect to q^2: -alpha exp(-beta q^2) (1 + beta q^2)/q^4."""
    R_ = -alpha * np.exp(-beta * x**2) * (1 + beta * x**2) / x**4
    return R_
