import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy import sparse as sparse
from scipy.sparse.linalg import eigs

h_pq = np.zeros((4,4))
s_pq = np.zeros((4,4))
Q_prqs = np.zeros((4,4,4,4))
F_pq = np.zeros((4,4))
deltaE_limit = 1e-5/27.211324570273 # In a.u.


alpha = np.array([0.297104, 1.236745, 5.749982, 38.216677]) # Given
C = np.array([1, 1, 1, 1]) # Starting values


# Chapter 4.3 in the book
# Algorithm at page 51

def Chi(r, p): # For wave function
    return np.exp(-p*r**2)
   
def Chi_sq(r, p, q): # Chi_p * Chi_q, for easier integration
    return np.exp(-(p+q)*r**2)

# Calculating h_pq
def h_eq(r, p,q): # Equation to integrate for H
    return r**2 * (3*q*Chi_sq(r, p,q) - 2*r**2*q**2*Chi_sq(r, p,q) - 2*Chi_sq(r, p,q)/r)
    

def H_pq(h_pq, alpha): # Definition from QM (integral)
    for p in range(4):
        for q in range(4):
            h_pq[p,q] = 4*np.pi*quad(h_eq, 0, np.inf, args=(alpha[p],alpha[q]))[0]
            # First element in the quad is the value and the other is \pm
    return h_pq

# Calculating S_pq
def s_eq(r, p, q): # Equation to integrate for S
    return r**2 * Chi_sq(r, p,q) # Chi_p * Chi_q


def S_pq(s_pq, alpha): # Definition from QM (integral)
    for p in range(4):
        for q in range(4):
            s_pq[p,q] = 4*np.pi*quad(s_eq, 0, np.inf, args=(alpha[p],alpha[q]))[0]
            # First element in the quad is the value and the other is \pm
    return s_pq


def Q_eq(Q_prqs, alpha): # Equation for Q
    for p in range(4):
        for r in range(4):
            for q in range(4):
                for s in range(4):
                    Q_prqs[p,r,q,s] = (2*np.pi**(5/2)) \
                        / ((alpha[p]+alpha[q]) * (alpha[r]+alpha[s]) \
                           * np.sqrt(alpha[p]+alpha[q]+alpha[r]+alpha[s]))
    return Q_prqs

def E_eq(Q, h, C): # Equation for calculating the energy
    E_G = 0
    for p in range(4):
        for q in range(4):
            E_G += 2*C[p]*C[q]*h[p,q]
            for r in range(4):
                for s in range(4):
                    E_G += Q[p,r,q,s]*C[p]*C[q]*C[r]*C[s]
    return E_G
    
    
# Calculate H, S and Q once and define a starting energy difference > deltaE_limit
H = H_pq(h_pq, alpha)
S = S_pq(s_pq, alpha)
Q = Q_eq(Q_prqs, alpha)
Ediff = 1


# Do calculations for new C vectors until Ediff > deltaE_limit
while Ediff > deltaE_limit:
    for p in range(4):
        for q in range(4):
            F_pq[p,q] = H[p,q]
            for r in range(4):
                for s in range(4):
                    F_pq[p,q] +=  Q[p,r,q,s]*C[r]*C[s]
            
    eig_val, eig_vec = eigh(F_pq, S) # Obtaining eigenvalues/vectors 
    
    min_index = np.argmin(eig_val) # Extracting index from lowest eigenvalue
    
    E_begin = E_eq(Q,H,C) # Energy from the beginning
    
    C = eig_vec[:,min_index] # New C is the vector corresponding to the lowest eigenvalue
        
    E = E_eq(Q, H, C) # Energy calculated with new C
    
    Ediff = np.abs(E - E_begin)
    
    
r_1 = np.linspace(0.00001,6,100)
Wave = np.zeros(len(r_1))
for i in range(len(r_1)):
    Wave[i] = np.sum(C * Chi(r_1[i], alpha))

# fig, ax = plt.subplots()

# ax.plot(r_1,Wave)
# ax.set_xlabel('r (Bohr radius)',fontsize = 15)
# ax.set_ylabel('$\phi$(r)',fontsize = 15)
# ax.xaxis.set_tick_params(labelsize=13)
# ax.yaxis.set_tick_params(labelsize=13)

# print('Task 1:')
# print('Ground state energy of helium atom in a.u.:', E)



# Task 2

  
def x_stuff(a, b, n):
    h = (b-a)/n
    x_i = np.zeros((n))
    for i in range(n):
        x_i[i] = a + i*h
    return x_i, h
    

def density(r, alpha, C, h): # n_s(r) = np.abs(phi(r))**2
    n_s = 4*np.exp(-2*r)*r
    
    
    return n_s


def Hatree_pot(r):
   # V = np.zeros(len(r))
    #for i in range(len(r)):
    V = 1/r - (1+1/r)*np.exp(-2*r)
    return V

# Solve with finite differences
a = 0.0000001
b = 100
n = 1000
h = x_stuff(a, b, n+1)[1]
rrr = x_stuff(a, b, n+1)[0]

#UU = -(rrr+1)*np.exp(-2*rrr)+1
U = np.zeros((n+1))
u_sqq = 4*rrr*np.exp(-2*rrr)
UUU = np.zeros((n+1))
UUU[0] = 0 ; UUU[1] = h

for i in range(1, n):
    UUU[i+1] = 2*UUU[i] - UUU[i-1] - h**2*u_sqq[i]


UUU = UUU / rrr
  
UUU[0] = 1
UUU[-1] = 0

# fig1, ax1 = plt.subplots()
# ax1.plot(rrr,UUU)
# #ax1.plot(rrr,UU/rrr)
# ax1.plot(rrr,Hatree_pot(rrr),'r--')
# ax1.legend(['FDM, U/r', 'Hatree, V$_{sH}$'])
# ax1.set_xlabel('r (Bohr radius)',fontsize = 15)
# ax1.set_ylabel('Potential [Hartree]',fontsize = 15)
# ax1.xaxis.set_tick_params(labelsize=13)
# ax1.yaxis.set_tick_params(labelsize=13)






# Task 3



def V_xc(r, wave): # Page 3 in the assignment sheet
    A = 0.0311
    B = -0.048
    C = 0.002
    D = -0.0116
    gamma = -0.1423
    beta_1 = 1.0529
    beta_2 = 0.3334
    N = len(r) -1
    n = n_eq(r, wave)
    eps_x = -3/4 * (3*n[:N]/np.pi)**(1/3)  
    eps_c = np.zeros((N))
    
    for i in range(N):
        # eps_c
        if r[i] >= 1:
            eps_c[i] = gamma/(1+beta_1*np.sqrt(r[i])+beta_2*r[i])
        else:
            eps_c[i] = A*np.log(r[i])+B+C*r[i]*np.log(r[i])+D*r[i]
            
    eps_xc = eps_x + eps_c
        # d/dn (eps_xc) = - n**(1/3)/4 * (3/np.pi)**(1/3)
    return eps_xc - (n[:N]**(1/3)/4 * (3/np.pi)**(1/3)), eps_xc


def n_eq(r, wave):
    #dr = r[1]-r[0]
   # integral = np.sum(wave**2*dr) # should be = 1
    #normalized = wave/np.sqrt(integral)
    return np.abs(wave)**2

def Wave_phi(u, r,n):
    phi = np.zeros((n))
    for i in range(n):
        phi[i] = u[i]/(r[i]*np.sqrt(4*np.pi))
    return phi


def E_calc(epsilon, wave, r, h):
    N = len(r) - 1
    E = np.zeros((N))
    dr = r[1] - r[0]
    for i in range(N):
        E[i] = 2*epsilon - np.sum(dr*(wave[i]**2) * \
                (0.5*Hatree_pot(r)[i] + V_xc(r,wave)[0][i] - \
                V_xc(r, wave)[1][i]))
            
    return E #2*np.sum(epsilon) - np.sum(h*wave**2 \
              #                           * (0.5*Hatree_pot(rrr) + V_xc(rrr,wave)[0] - \
              #                              V_xc(rrr, wave)[1]))


def KS_eq(a, b, n): # Kohn-Sham eq.
    h = x_stuff(a, b, n+1)[1]
    rrr = x_stuff(a, b, n+1)[0]
    A = np.zeros((n, n))
    
    #A[0, 1] = -1/(2*h**2)
    #A[n, n-1] = -1/(2*h**2)
    #u = np.zeros((n+1))
    #limit = 1e-6 # a.u.
    diff = 1
    E = 1
    phi = 4*np.exp(-2*rrr)
    
    
    #while diff > limit:
    K = -(0.5/h**2)*(np.diag(np.ones(n-1),-1) - np.diag(2*np.ones(n),0) + np.diag(np.ones(n-1),1))

    for i in range(n):
         #A[i, i-1] = -1/(2*h**2)
         A[i, i] = -2/rrr[i] + Hatree_pot(rrr)[i] + V_xc(rrr,phi)[0][i]
         #A[i, i+1] = -1/(2*h**2)
         
         
    #A[0, 0] = (-2/rrr[0] + Hatree_pot(rrr)[0] + V_xc(rrr,phi)[0][0])
    #A[n, n] = (-2/rrr[n] + Hatree_pot(rrr)[n] + V_xc(rrr,phi)[0][n])
    
         
    Hse = K + A
     #B = np.identity(n+1)
    eigvalu, eigvect = eigh(Hse) # Solve eigen problem
     
     #epsilon = eigvalu/eigvalu[0]
    eigvect = np.transpose(eigvect)
    u = eigvect[1,:]
    u[-1] = 0
    u[0] = 1
     
     #print(epsilon.shape)
    # print(eigvect.shape)
     #phi = Wave_phi(u[index,:],rrr,n+1)
    E_0 = E_calc(eigvalu[1], u, rrr, h)
     
    diff = np.abs(eigvalu[1] - E)
    E = eigvalu[1]
     
        #print('E',E)
    #E_0 = 0

    return eigvalu[1:], eigvect, E_0, phi

n=100
b = 10

rrr = np.linspace(a,b,n)
KS = KS_eq(a, b, n)
print('Ground state energy: ',KS[0])




def Schr_eq(r): # Schrödinger eq.
    n = len(r) - 1
    A = np.zeros((n+1, n+1))
    h = x_stuff(np.min(r), np.max(r), n+1)[1]
    A[0, 0] = -2/(2*h**2) - 2/r[0]
    A[n, n] = -2/(2*h**2) - 2/r[n]
    #A[0, 1] = 1/(2*h**2)
    #A[n, n-1] = 1/(2*h**2)
    for i in range(1, n):
        A[i, i-1] = 1/(2*h**2)
        A[i, i] = -2/(2*h**2) - 2/r[i]
        A[i, i+1] = 1/(2*h**2)
        
    val, vec = eigh(-A)
    
    #val = val/val[0]
    
    vec = np.transpose(vec) # Wave functions, row=state, col=points
    return val, vec

print(Schr_eq(rrr)[1])


fig2, ax2 = plt.subplots()
#
ax2.plot(rrr,KS[1][2,:])
ax2.plot(rrr,Schr_eq(rrr)[1][1,:],'r--')
#
ax2.legend(['DFT','Schr'])
ax2.set_xlabel('r (Bohr radius)',fontsize = 15)
ax2.set_ylabel('$\phi$(r)',fontsize = 15)
ax2.xaxis.set_tick_params(labelsize=12)
ax2.yaxis.set_tick_params(labelsize=12)

# Calculate the energy







