import numpy as np

def lml(alpha, beta, Phi, Y):

    N_val = -0.5*Y.shape[0]
    
    TwopiN2 = np.power(2*np.pi, N_val)
    
    PhiAlphaPhiT = np.matmul(Phi,alpha*np.transpose(Phi))
    A = PhiAlphaPhiT + beta*np.identity(len(PhiAlphaPhiT))
    Apoint5 = np.linalg.det(A)**(-0.5)
    exp_term = np.exp(-0.5* np.matmul( np.matmul(np.transpose(Y),np.linalg.inv(A)),Y))
    
    lml = np.asscalar(TwopiN2 * Apoint5 *  exp_term)
    
    return np.log(lml)

def grad_lml(alpha, beta, Phi, Y):
    
    N_val = -0.5*Y.shape[0]
    
    TwopiN2 = np.power(2*np.pi, N_val)
    
    PhiAlphaPhiT = np.matmul(Phi,alpha*np.transpose(Phi))
    
    A = PhiAlphaPhiT + beta*np.identity(len(PhiAlphaPhiT))
    Apoint5 = np.linalg.det(A)**(-0.5)
    
    exp_term = np.exp(-0.5* np.matmul( np.matmul(np.transpose(Y),np.linalg.inv(A)),Y))
    
    lml = np.asscalar(TwopiN2 * Apoint5 *  exp_term)

    phiphiT = np.matmul(Phi,np.transpose(Phi))
    
    Var1 = np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(Y),np.linalg.inv(A)), phiphiT),np.linalg.inv(A)),Y)
    Var2 = np.matmul(np.matmul(np.matmul(np.transpose(Y),np.linalg.inv(A)),np.linalg.inv(A)),Y)
    
    Var = np.power(lml, -1)*TwopiN2*0.5*exp_term*Apoint5
    
    grad_alpha = Var * (Var1 - np.trace(np.matmul(np.linalg.inv(A),phiphiT)))
    
    grad_beta  = Var * (Var2 - np.trace(np.linalg.inv(A)))
   
    return np.array([grad_alpha[0][0],grad_beta[0][0]])


