#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 22:01:10 2018

@author: FeiChenyi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 19:55:14 2018

@author: Fei Chenyi
"""

import random
# fft in numpy
import numpy as np
# plotting tools
import matplotlib.pyplot as plt
# switch to background plotting
plt.switch_backend('agg')
# constant Pi
from math import pi
import os

def hat2producthat(ahat,bhat):
    ainv = np.fft.ifft2(ahat);
    binv = np.fft.ifft2(bhat);
    ab = ainv*binv;
    return np.fft.ifft2(ab)
    
myname = "v0"
trial = 1
directory = "von_karman_data/"+ myname + "/" + str(trial) +"/"
if not os.path.exists(directory):
    os.makedirs(directory)
filew = open(directory+"w_"+str(trial)+".txt",'w')

#==============================================================================
# Numerics
#==============================================================================
if __name__ == '__main__':
    # geometries
    dt = 0.10; dx = 0.1;
    N = 512; M = 2; d = 1;
    
    # elastic parameter 
    Ef = 1000.0
    nuf = 0.3
    Es = 1.0
    nus = 0.3
    h = 0.01
    
    # damping parameter 
    xi = 0.3*Es/h
    ERROR_CONV = 1e-6
    # noises
#    Fu = 1.0e-6; stdNoise = np.sqrt(2.0*Fu*dt/dx**d);
    # interaction
#    chi = 3.50; chiMat = create_equal_chiMat(M, chi);
    
    # spatial points and wave numbers
    x = dx * np.arange(N)
    kx = 2*pi*np.fft.fftfreq(x.shape[-1])*(1.0/dx)
    kx = kx.reshape([N,1])
    ky = 2*pi*np.fft.fftfreq(x.shape[-1])*(1.0/dx)
    k2 = kx*kx + ky*ky
    k = np.sqrt(k2)
    k_ = k
    k_[0,0] = 1e-8
    
    # define Dmatrix
    D0 = Es*(1-nus)/(6-8*nus)
    D11 = D0 * (4*k2*(1-nus)-ky*ky) / k_
    D12 = D0 * kx*ky/k_
    D13 = D0 * (2*1j)*(1-2*nus) * kx
    
    D21 = D0 * kx*ky/k_
    D22 = D0 * (4*k2*(1-nus)-kx*kx) / k_
    D23 = D0 * (2*1j)*(1-2*nus) * ky
    
    D31 = D0*(-2*1j)*(1-2*nus)*kx
    D32 = D0*(-2*1j)*(1-2*nus)*ky
    D33 = D0*4.*(1-nus)*np.sqrt(k2)
    
    # define C matrix
    C11 = 2*D11 + Ef*h*((1-nuf)*k2 + (1+nuf)*kx*kx)
    C12 = 2*D12 + Ef*h*((1+nuf)*kx*ky)
    C21 = 2*D21 + Ef*h*((1+nuf)*kx*ky)
    C22 = 2*D22 + Ef*h*((1-nuf)*k2 + (1+nuf)*ky*ky)
    Jc = C11*C22-C12*C21
    Jc_ = Jc
    Jc_[0,0] = 1e-8
    invC11 = C22/Jc_
    invC12 = -C12/Jc_
    invC21 = -C21/Jc_
    invC22 = C11/Jc_
    
    # initialize - displacement
    u1_ = np.zeros([N,N])
    u2_ = np.zeros([N,N])
#    w_  = np.zeros([N,N])
    w_ = np.random.normal(0.0,0.001*h,[N,N])
    
    # initialize - strain
    e11_init = 0.1;
    e22_init = 0.1;
    e12_init = 0.0;
    eps11i_ = e11_init * np.ones([N,N])
    eps22i_ = e22_init * np.ones([N,N])
    eps12i_ = e12_init * np.ones([N,N])
    eps21i_ = e12_init * np.ones([N,N])
    e1ggihat = 1j*kx*np.fft.fft2(eps11i_) + 1j*ky*np.fft.fft2(eps12i_)
    e2ggihat = 1j*kx*np.fft.fft2(eps21i_) + 1j*ky*np.fft.fft2(eps22i_)
    egg1ihat = 1j*kx*np.fft.fft2(eps11i_) + 1j*kx*np.fft.fft2(eps22i_) 
    egg2ihat = 1j*ky*np.fft.fft2(eps11i_) + 1j*ky*np.fft.fft2(eps22i_) 
    
    # loop until np.linalg.norm
    error = 1.
    # calculate fft of u and w 
    u1hat = np.fft.fft2(u1_)
    u2hat = np.fft.fft2(u2_)
    what = np.fft.fft2(w_)
    
        
    while error > ERROR_CONV:
        
        u1hat_ =u1hat
        u2hat_ = u2hat
        what_ = what
        
        # calculate fft of u_,i and w_,i
        u11hat_ = 1.j*kx*u1hat_
        u12hat_ = 1.j*ky*u1hat_
        u21hat_ = 1.j*kx*u2hat_
        u22hat_ = 1.j*ky*u2hat_
        
        w1hat_ = 1.j*kx*what_
        w2hat_ = 1.j*ky*what_
        
        # calculate fft of eps_ij
        eps11hat_ = np.fft.fft2(eps11i_) + u11hat_ + 0.5*hat2producthat(w1hat_,w1hat_)
        eps12hat_ = np.fft.fft2(eps12i_) + 0.5*(u12hat_+u21hat_) + 0.5*hat2producthat(w1hat_,w2hat_)
    #    eps21hat_ = np.fft.fft2(eps21i_) + 0.5*(u12hat_+u21hat_) + 0.5*hat2producthat(w1hat_,w2hat_)
        eps22hat_ = np.fft.fft2(eps22i_) + u22hat_ + 0.5*hat2producthat(w2hat_,w2hat_)
        
        # calculate fft of N_ij
        N11hat_ = Ef*h*(eps11hat_ + nuf*eps22hat_)
        N22hat_ = Ef*h*(eps22hat_ + nuf*eps11hat_)
        N12hat_ = Ef*h*(1-nuf)*eps12hat_
        N21hat_ = Ef*h*(1-nuf)*eps12hat_
        
        # update what according to equation (27)
        Nw1hat_ = hat2producthat(N11hat_,w1hat_) + hat2producthat(N21hat_,w2hat_)
        Nw2hat_ = hat2producthat(N12hat_,w1hat_) + hat2producthat(N22hat_,w2hat_)
        what = 1/(D33 + Ef*h**3*k2*k2 + xi)*(1j*kx*Nw1hat_ + 1j*ky*Nw2hat_ - D31*u1hat_ - D32*u2hat_ + xi*what_ )
        
        # update u1 and u2 according to equation (23-25)
        w1hat = 1j*kx*what
        w2hat = 1j*ky*what
        w11hat = -kx*kx*what
        w12hat = -kx*ky*what
        w21hat = -kx*ky*what
        w22hat = -ky*ky*what
        
        w1gwghat = hat2producthat(w11hat,w1hat) + hat2producthat(w12hat,w2hat)
        w1wgghat = hat2producthat(w1hat,w11hat) + hat2producthat(w1hat,w22hat)
        w2gwghat = hat2producthat(w21hat,w1hat) + hat2producthat(w22hat,w2hat)
        w2wgghat = hat2producthat(w2hat,w11hat) + hat2producthat(w2hat,w22hat)
        
        b1hat = -2*D13*what + Ef*h*((1+nuf)*(w1gwghat) + (1-nuf)*(w1wgghat) + 2*(1-nuf)*(e1ggihat) + 2*nuf*(egg1ihat));
        b2hat = -2*D23*what + Ef*h*((1+nuf)*(w2gwghat) + (1-nuf)*(w2wgghat) + 2*(1-nuf)*(e2ggihat) + 2*nuf*(egg2ihat));
        u1hat = invC11 * b1hat + invC12 * b2hat;
        u2hat = invC21 * b1hat + invC22 * b2hat;
        
        error = np.linalg.norm(what - what_)/np.sqrt(N)
        print(error)

w = np.fft.ifft2(what)

for i in np.arange(N):
    for j in np.arange(N):
        filew.write(str(w[i,j])+' ')
    filew.write('\n')

filew.close()