#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:53:10 2020

@author: salonisaxena
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def cdotu(c,ux,uy):
    c_dot_u = np.zeros((N,N,9))
    for i in range(9):
        c_dot_u[:,:,i] = c[0,i]*ux+c[1,i]*uy
    return c_dot_u

def cu_sq(c,ux,uy):
    return cdotu(c,ux,uy)**2

def usq(ux,uy):
    return ux**2+uy**2

def cdotF(c,Fx,Fy):
    c_dot_F = np.zeros((N,N,9))
    for i in range(9):
        c_dot_F[:,:,i] = c[0,i]*Fx+c[1,i]*Fy
    return c_dot_F

def udotF(ux,uy,Fx,Fy):
    return ux*Fx + uy*Fy

def equilibrium(c,ux,uy,rho):
    feq = np.zeros((N,N,9))
    for i in range(9):
        #feq[:,:,i] = w[i]*rho*(1 + 3*cdotu(c,ux,uy)[:,:,i] + 4.5*cu_sq(c,ux,uy)[:,:,i] - 1.5*usq(ux,uy))
        feq[:,:,i] = w[i]*(rho + 3*cdotu(c,ux,uy)[:,:,i] + 4.5*cu_sq(c,ux,uy)[:,:,i] - 1.5*usq(ux,uy))

    return feq
    
def F_lb(c,ux,uy,Fx,Fy):
    discreteF = np.zeros((N,N,9))
    for i in range(9):
        discreteF[:,:,i] = w[i]*(3*cdotF(c,Fx,Fy)[:,:,i]) #+ 9*cdotF(c,Fx,Fy)[:,:,i]*cdotu(c,ux,uy)[:,:,i] - 3*udotF(ux,uy,Fx,Fy))
    return discreteF   #

def collision(f,feq):
    fstar = np.zeros((N,N,9))
    for i in range(9):
        fstar[:,:,i] = f[:,:,i]*(1-dt*tau_inv)+feq[:,:,i]*dt*tau_inv + dt*(1-1/(2*tau_lbm))*F_disc[:,:,i]
    return fstar

def laplacian(f):  #f is NxN matrix
    lapx = np.gradient(np.gradient(f,axis=1),axis=1)
    lapy = np.gradient(np.gradient(f,axis=0),axis=0)
    return lapx + lapy

def ddx(f): #x derivative of f at each (y,x)
    return np.gradient(f,axis=1)
    
def ddy(f): #y derivative of f at each (y,x)
    return np.gradient(f,axis=0) 

def D(vel_grad):
    d = np.zeros((N,N,2,2))
    for i in range(N):
        for j in range(N):
            d[i,j] = (vel_grad[i,j]+np.transpose(vel_grad[i,j]))/2
    return d

def omega(vel_grad):
    om = np.zeros((N,N,2,2))
    for i in range(N):
        for j in range(N):
            om[i,j] = (vel_grad[i,j]-np.transpose(vel_grad[i,j]))/2
    return om

def W(ux,uy):
    w = np.zeros((N,N,3))
    w[:,:,0] = ddx(ux)
    w[:,:,1] = ddy(ux)
    w[:,:,2] = ddx(uy)
    return w
     
def stress_sym(Q): #use bulk values of Q                               
    sigma = np.zeros((N,N,3))  #sig_xx, sig_xy
    h = H(Q)
    sigma[:,:,0] = -2*xi*(h[:,:,0]*Q[:,:,0]+h[:,:,1]*Q[:,:,1]) + 4*(Q[:,:,0]*h[:,:,0]+Q[:,:,1]*h[:,:,1])*xi*(Q[:,:,0]+0.5) - xi*h[:,:,0] - 4*ddx(Q[:,:,0])**2 - 4*ddx(Q[:,:,1])**2    #sigma xx
    sigma[:,:,1] =  4*(Q[:,:,0]*h[:,:,0]+Q[:,:,1]*h[:,:,1])*xi*Q[:,:,1] - 1*0.5*xi*h[:,:,1] - 4*ddy(Q[:,:,0])*ddx(Q[:,:,0]) - 4*ddy(Q[:,:,1])*ddx(Q[:,:,1])
    sigma[:,:,2] = -2*xi*(h[:,:,0]*Q[:,:,0]+h[:,:,1]*Q[:,:,1]) + 4*(Q[:,:,0]*h[:,:,0]+Q[:,:,1]*h[:,:,1])*xi*(-Q[:,:,0]+0.5) + xi*h[:,:,0] - 4*ddy(Q[:,:,0])**2 - 4*ddy(Q[:,:,1])**2
    return sigma

def stress_antisym(Q):
    h = H(Q)
    tau = np.zeros((N,N))
    tau = 2*(Q[:,:,0]*h[:,:,1]-Q[:,:,1]*h[:,:,0])  #tau xy = -tau yx, diags 0
    return tau

def force(Q):
    Fo = np.zeros((N,N,2)) #2 force components at each x,y
    temp3 = stress_sym(Q)
    temp4 = stress_antisym(Q)
    Fo[:,:,0] = ddx(temp3[:,:,0]) + ddy(temp3[:,:,1]) + ddy(temp4)
    Fo[:,:,1] = ddx(temp3[:,:,1]) + ddx(-temp4) + ddy(temp3[:,:,2])
    return Fo[:,:,0], Fo[:,:,1]
    

def relax_nondim(Q):
    qre = np.zeros(np.shape(Q))
    qre[:,:,0] = (-1*2*(1-0.5*u)*Q[:,:,0] - 1*4*u*Q[:,:,0]**3 - 1*4*u*Q[:,:,0]*Q[:,:,1]**2)/(Er*l**2) + 1*laplacian(Q[:,:,0])/Er
    qre[:,:,1] = (-1*2*(1-0.5*u)*Q[:,:,1] - 1*4*u*Q[:,:,1]**3 - 1*4*u*Q[:,:,1]*Q[:,:,0]**2)/(Er*l**2) + 1*laplacian(Q[:,:,1])/Er
    return qre

def H(Q):
    h = np.zeros(np.shape(Q))
    h[:,:,0] = (-1*2*(1-0.5*u)*Q[:,:,0] - 1*4*u*Q[:,:,0]**3 - 1*4*u*Q[:,:,0]*Q[:,:,1]**2)/(l**2) + 1*laplacian(Q[:,:,0])
    h[:,:,1] = (-1*2*(1-0.5*u)*Q[:,:,1] - 1*4*u*Q[:,:,1]**3 - 1*4*u*Q[:,:,1]*Q[:,:,0]**2)/(l**2) + 1*laplacian(Q[:,:,1])
    return h
    
def advec_nondim(Q):
    qad = np.zeros(np.shape(Q))
    qad[:,:,0] = -ux*ddx(Q[:,:,0]) - uy*ddy(Q[:,:,0]) 
    qad[:,:,1] = -ux*ddx(Q[:,:,1]) - uy*ddy(Q[:,:,1]) 
    return qad    

def S(ux,uy,Q):
    s = np.zeros(np.shape(Q))
    vel_grad = W(ux,uy)
    tr = (2*Q[:,:,0]*vel_grad[:,:,0]) + Q[:,:,1]*(vel_grad[:,:,1]+vel_grad[:,:,2])
    
    s[:,:,0] = xi*(2*Q[:,:,0]*vel_grad[:,:,0] + Q[:,:,1]*(vel_grad[:,:,1]+vel_grad[:,:,2])) + xi*vel_grad[:,:,0] + Q[:,:,1]*(vel_grad[:,:,1]-vel_grad[:,:,2]) - 2*xi*tr*(Q[:,:,0]+0.5)
    s[:,:,1] = 0.5*xi*(vel_grad[:,:,1]+vel_grad[:,:,2]) - Q[:,:,0]*(vel_grad[:,:,1]-vel_grad[:,:,2]) - 2*xi*tr*Q[:,:,1]
    return s
 

def eigens(Q):
    eigval = np.sqrt(Q[:,:,0]**2 + Q[:,:,1]**2)
    n1sq = 0.5 + 0.5*(Q[:,:,0]/np.sqrt(Q[:,:,0]**2 + Q[:,:,1]**2))
    n2sq = 1 - n1sq
    return eigval, np.sqrt(n1sq), np.sqrt(n2sq)
  
"""
Bottom, left, right walls are situated halfway between fluid and solid nodes. 
Top wall is situated on fluid nodes. BCs for Q are imposed on the walls. Bounceback imposed on fluid nodes."""

N = 128   #256 
dx = 1
x = np.arange(0,N)+0.5
y = np.arange(0,N)+0.5
X,Y = np.meshgrid(x,y)

scale = 1 #lattice space in nm


#LC parameters
u = 6
xi = 0.7

Q = np.zeros((N,N,2))
nx = 0.8
ny = np.sqrt(1-nx**2)
q = 1e-4 #scalar order parameter

Q[:,:,0] = q*(nx**2-0.5)
Q[:,:,1] = q*nx*ny

#BCs on Q
nx_top = 0
ny_top = 1-nx_top**2

nx_left = 1
ny_left = 1-nx_left**2
#top
Q[N-1,:,0] = q*(nx_top**2-0.5)
Q[N-1,:,1] = q*nx_top*ny_top
Q[0,:,0] = q*(nx_top**2-0.5)
Q[0,:,1] = q*nx_top*ny_top
#left, right
Q[1:N-1,0,0] = q*(nx_left**2-0.5)
Q[1:N-1,0,1] = q*nx_left*ny_left
Q[1:N-1,N-1,0] = q*(nx_left**2-0.5)
Q[1:N-1,N-1,1] = q*nx_left*ny_left



#LBM
#D2Q9 lattice
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]) # weights
c = np.array([[0, 1, 0, -1,  0, 1, -1, -1,  1],  # velocities, x components
              [0, 0, 1,  0, -1, 1,  1, -1, -1]]) # velocities, y components

tau_lbm = 0.9
tau_inv = 1/tau_lbm
nu = (2*tau_lbm - 1)/6
Re = 0.09
u_max = nu*Re/N
Er = 100
l = 20
dt = 1

#define initial physical quantities

rho = np.ones((N,N))
ux = np.zeros((N,N))
uy = np.zeros((N,N))
ux[N-1] = u_max
#Forces
Fx = 1*force(Q)[0]  #full domain
Fy = 1*force(Q)[1]

f_old = equilibrium(c,-Fx/2,-Fy/2,rho)
feq = f_old
f_new= np.zeros((N,N,9))
F_disc = F_lb(c,ux,uy,Fx,Fy)

t = 0
n=0
start = time.time()

#%%
T = 170000 #time of integration\
start = time.time()
while t < T:
    
    f_coll = collision(f_old,feq)
    for i in range(9): #periodic streaming
        f_new[:,:,i] = np.roll(np.roll(f_coll[:,:,i],c[0,i],axis=1), c[1,i], axis=0)
        

    #Boundary conditions
    
    #Bottom wall, except corners, need f2, f5, f6
    f_new[0,1:N-1,2] = f_coll[0,1:N-1,4]
    f_new[0,1:N-1,5] = f_coll[0,1:N-1,7]
    f_new[0,1:N-1,6] = f_coll[0,1:N-1,8]

    #left wall, except corners, need f1,f5, f8
    f_new[1:N-1,0,1] = f_coll[1:N-1,0,3]
    f_new[1:N-1,0,5] = f_coll[1:N-1,0,7]
    f_new[1:N-1,0,8] = f_coll[1:N-1,0,6]

    #right wall, except corners, need f3, f6, f7
    f_new[1:N-1,N-1,3] = f_coll[1:N-1,N-1,1]
    f_new[1:N-1,N-1,6] = f_coll[1:N-1,N-1,8]
    f_new[1:N-1,N-1,7] = f_coll[1:N-1,N-1,5]
    
    #top wall, nebb, except corners
    
    f_new[N-1,1:N-1,4] = f_new[N-1,1:N-1,2] - (2/3)*rho[N-1,1:N-1]*uy[N-1,1:N-1] + (Fy[N-1,1:N-1]/6)
    f_new[N-1,1:N-1,7] = f_new[N-1,1:N-1,5] + 0.5*(f_new[N-1,1:N-1,1]-f_new[N-1,1:N-1,3]) - 0.5*rho[N-1,1:N-1]*ux[N-1,1:N-1] - (1/6)*rho[N-1,1:N-1]*uy[N-1,1:N-1] + 0.25*Fx[N-1,1:N-1] + Fy[N-1,1:N-1]/6
    f_new[N-1,1:N-1,8] = f_new[N-1,1:N-1,6] - 0.5*(f_new[N-1,1:N-1,1]-f_new[N-1,1:N-1,3]) + 0.5*rho[N-1,1:N-1]*ux[N-1,1:N-1] - (1/6)*rho[N-1,1:N-1]*uy[N-1,1:N-1] - 0.25*Fx[N-1,1:N-1] + Fy[N-1,1:N-1]/6
    
    #corners
    #bottom left, standard bb
    f_new[0,0,1] = f_coll[0,0,3]
    f_new[0,0,2] = f_coll[0,0,4]
    f_new[0,0,5] = f_coll[0,0,7]
    f_new[0,0,6] = f_coll[0,0,8]
    f_new[0,0,8] = f_coll[0,0,6]

    #bottom right, standard bb
    f_new[0,N-1,3] = f_coll[0,N-1,1]
    f_new[0,N-1,2] = f_coll[0,N-1,4]
    f_new[0,N-1,5] = f_coll[0,N-1,7]
    f_new[0,N-1,7] = f_coll[0,N-1,5]
    f_new[0,N-1,6] = f_coll[0,N-1,8]
    
    #top left, nebb
    f_new[N-1,0,1] = f_new[N-1,0,3] + (2/3)*ux[N-1,0]*rho[N-1,0]
    f_new[N-1,0,4] = f_new[N-1,0,2] - (2/3)*rho[N-1,0]*uy[N-1,0]
    f_new[N-1,0,8] = f_new[N-1,0,6] + (1/6)*rho[N-1,0]*ux[N-1,0]
    f_new[N-1,0,5] = (1/12)*rho[N-1,0]*ux[N-1,0]
    f_new[N-1,0,7] = -(1/12)*rho[N-1,0]*ux[N-1,0]
    f_new[N-1,0,0] = rho[N-1,0]-f_new[N-1,0,1]-f_new[N-1,0,2]-f_new[N-1,0,3]-f_new[N-1,0,4]-f_new[N-1,0,5]-f_new[N-1,0,6]-f_new[N-1,0,7]-f_new[N-1,0,8]
    
    #top right, nebb
    f_new[N-1,N-1,3] = f_new[N-1,N-1,1] - (2/3)*ux[N-1,N-1]*rho[N-1,N-1]
    f_new[N-1,N-1,4] = f_new[N-1,N-1,2] - (2/3)*rho[N-1,N-1]*uy[N-1,N-1]
    f_new[N-1,N-1,7] = f_new[N-1,N-1,5] - (1/6)*rho[N-1,N-1]*ux[N-1,N-1]
    f_new[N-1,N-1,8] = (1/12)*rho[N-1,N-1]*ux[N-1,N-1]
    f_new[N-1,N-1,6] = -(1/12)*rho[N-1,N-1]*ux[N-1,N-1]
    f_new[N-1,N-1,0] = rho[N-1,0]-f_new[N-1,N-1,1]-f_new[N-1,N-1,2]-f_new[N-1,N-1,3]-f_new[N-1,N-1,4]-f_new[N-1,N-1,5]-f_new[N-1,N-1,6]-f_new[N-1,N-1,7]-f_new[N-1,N-1,8]
    
    #update rho, u
    rho = np.sum(f_new,axis=2)
    ux = np.sum(np.multiply(c[0],f_new),axis=2) + 0.5*Fx     #/rho
    uy = np.sum(np.multiply(c[1],f_new),axis=2) + 0.5*Fy     #/rho
    
    #set rho, u on moving wall
    #top
    rho[N-1] = (1/(1+uy[N-1]))*(f_new[N-1,:,0]+f_new[N-1,:,1]+f_new[N-1,:,3] + 2*(f_new[N-1,:,2]+f_new[N-1,:,5]+f_new[N-1,:,6]) + 0.5*Fy[N-1])
    ux[N-1] = u_max
    uy[N-1] = 0
    #find new equilibrium
    feq = equilibrium(c,ux,uy,rho)
    
    #LC dynamics
    Q_new = Q + dt*(relax_nondim(Q) + 1*advec_nondim(Q) + 1*S(ux,uy,Q))   #use full velcoity domain for Q evolution
    
    qt = 2*np.sqrt(Q_new[N-2,:,0]**2 + Q_new[N-2,:,1]**2)
    qb = 2*np.sqrt(Q_new[1,:,0]**2 + Q_new[1,:,1]**2)
    ql = 2*np.sqrt(Q_new[:,1,0]**2 + Q_new[:,1,1]**2)
    qr = 2*np.sqrt(Q_new[:,N-2,0]**2 + Q_new[:,N-2,1]**2)
    
    #top
    Q_new[N-1,:,0] = qt*(nx_top**2-0.5)
    Q_new[N-1,:,1] = qt*nx_top*ny_top
    Q_new[0,:,0] = qb*(nx_top**2-0.5)
    Q_new[0,:,1] = qb*nx_top*ny_top
    #left, right
    Q_new[:,0,0] = ql*(nx_left**2-0.5)
    Q_new[:,0,1] = ql*nx_left*ny_left
    Q_new[:,N-1,0] = qr*(nx_left**2-0.5)
    Q_new[:,N-1,1] = qr*nx_left*ny_left
    
    #Update forces using bulk values of Q
    Fx = 1*force(Q_new)[0]
    Fy = 1*force(Q_new)[1]
    F_disc = F_lb(c,ux,uy,Fx,Fy)
    
    #loop back
    f_old = f_new
    Q = Q_new
    if t%100==0:
        print(t, np.max(Fx), np.max(Fy), np.max(np.sqrt(Q_new[:,:,0]**2 + Q_new[:,:,1]**2)))
        #print(t, np.max(relax_nondim(Q)[:,:,0])/gamma, np.max(advec_nondim(Q)[:,:,0]), np.max(S(ux,uy,Q)[:,:,0]))
    t += dt
   
print(time.time()-start,np.sqrt(Q_new[53,37,0]**2 + Q_new[53,37,1]**2))
#%%
Z = np.sqrt(usq(ux,uy))
plt.figure()    
plt.contourf(X, Y, Z, 80, cmap='RdYlBu')
plt.colorbar()

fig, ax = plt.subplots()
ax.streamplot(X,Y,ux,uy, density = 2)
plt.show()
#%%
plt.figure()    
plt.contourf(X, Y, 2*np.sqrt(Q[:,:,0]**2 + Q[:,:,1]**2), 50, cmap='RdYlBu')
plt.colorbar()
plt.quiver(X, Y, eigens(Q)[1], eigens(Q)[2], units='xy', headlength=0, headaxislength=0)

#%%
plt.figure()    
plt.contourf(X, Y, 0*gamma*relax_nondim(Q)[:,:,0]/gamma +1*advec_nondim(Q)[:,:,0] + 1*S(ux,uy,Q)[:,:,0], 25, cmap='RdYlBu')
plt.colorbar()
#%%
plt.figure()  
plt.title('Q_xx')  
plt.contourf(X, Y, Q[:,:,0], 25, cmap='RdYlBu')
plt.colorbar()

plt.figure()  
plt.title('Q_xy')  
plt.contourf(X, Y, Q[:,:,1], 25, cmap='RdYlBu')
plt.colorbar()
#%%
plt.figure()  
plt.title('ddx Q_xx')  
plt.contourf(X, Y, ddx(Q[:,:,0]), 25, cmap='RdYlBu')
plt.colorbar()

plt.figure()  
plt.title('ddx Q_xy')  
plt.contourf(X, Y, ddx(Q[:,:,1]), 25, cmap='RdYlBu')
plt.colorbar()

plt.figure()  
plt.title('ddy Q_xx')  
plt.contourf(X, Y, ddy(Q[:,:,0]), 25, cmap='RdYlBu')
plt.colorbar()

plt.figure()  
plt.title('ddy Q_xy')  
plt.contourf(X, Y, ddy(Q[:,:,1]), 25, cmap='RdYlBu')
plt.colorbar()
#%%
plt.figure()  
plt.title('sigma_xx')  
plt.contourf(X, Y, (stress_sym(Q)[:,:,0]), 25, cmap='RdYlBu')
plt.colorbar()

plt.figure()  
plt.title('ddx(sigma_xx)')  
plt.contourf(X, Y, ddx(stress_sym(Q)[:,:,0]), 25, cmap='RdYlBu')
plt.colorbar()

plt.figure()  
plt.title('ddy(sigma_xx)')  
plt.contourf(X, Y, ddy(stress_sym(Q)[:,:,0]), 25, cmap='RdYlBu')
plt.colorbar()
#%%
plt.figure() 
plt.title('sigma_xy')   
plt.contourf(Xf, Yf, stress_sym(Qb)[:,:,1], 25, cmap='RdYlBu')
plt.colorbar()

plt.figure()  
plt.title('ddx(sigma_xy)')  
plt.contourf(Xf, Yf, ddx(stress_sym(Qb)[:,:,1]), 25, cmap='RdYlBu')
plt.colorbar()

plt.figure()  
plt.title('ddy(sigma_xy)')  
plt.contourf(Xf, Yf, ddy(stress_sym(Qb)[:,:,1]), 25, cmap='RdYlBu')
plt.colorbar()
#%%
plt.figure()   
plt.title('sigma_yy') 
plt.contourf(Xf, Yf, stress_sym(Qb)[:,:,2], 25, cmap='RdYlBu')
plt.colorbar()

plt.title('ddx(sigma_yy)')  
plt.contourf(Xf, Yf, ddx(stress_sym(Qb)[:,:,2]), 25, cmap='RdYlBu')
plt.colorbar()

plt.figure()  
plt.title('ddy(sigma_yy)')  
plt.contourf(Xf, Yf, ddy(stress_sym(Qb)[:,:,2]), 25, cmap='RdYlBu')
plt.colorbar()
#%%
plt.figure()    
plt.title('tau_xy')
plt.contourf(Xf, Yf, stress_antisym(Qb), 25, cmap='RdYlBu')
plt.colorbar()

#%%
plt.figure()  
plt.title('Fx')  
plt.contourf(X, Y, Fx, 25, cmap='RdYlBu')
plt.colorbar()

plt.figure()  
plt.title('Fy')  
plt.contourf(X, Y, Fy, 25, cmap='RdYlBu')
plt.colorbar()
#Q_full[11,11]
#array([ 0.10879041, -0.26998959])
#for u=3, used gamma=1, a=L1=0.01 0.28217917980845414
#for u=3.5, a=1e-4/9, gamma=0.1/9, L1=1e-3/81
#%%
m1 = np.zeros((2,2)) #w
m1[0,0] = np.random.randint(-2,2)
m1[0,1] = np.random.randint(-3,3)
m1[1,0] = np.random.randint(1,4)
m1[1,1] = -m1[0,0]
sym = 0.5*(m1+np.transpose(m1))
antisym = 0.5*(m1-np.transpose(m1))

m2 = np.zeros((2,2)) #w
m2[0,0] = np.random.randint(-2,2)
m2[0,1] = np.random.randint(-3,3)
m2[1,0] = m2[0,1]
m2[1,1] = -m2[0,0] #q
Tr = np.trace(np.matmul(m2,m1))
print(np.matmul(xi*sym + antisym, m2+0.5*np.eye(2)) + np.matmul(m2+0.5*np.eye(2), xi*sym-antisym) - 2*xi*Tr*(m2+0.5*np.eye(2)))
#%%
v_grad = np.zeros((N,N,3))
v_grad[:,:,0] = m1[0,0]
v_grad[:,:,1] = m1[0,1]
v_grad[:,:,2] = m1[1,0]

Q1 = np.zeros((N,N,2))
Q1[:,:,0] = m2[0,0]
Q1[:,:,1] = m2[0,1]

print(S(v_grad,Q1))