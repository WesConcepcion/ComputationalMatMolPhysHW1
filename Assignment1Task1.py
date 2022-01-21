import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import quad

h_pq = np.zeros((4,4))
s_pq = np.zeros((4,4))
Q_prqs = np.zeros((4,4,4,4))
F_pq = np.zeros((4,4))
deltaE_limit = 1e-5/27.211324570273 # In a.u.


alpha = np.array([0.297104, 1.236745, 5.749982, 38.216677]) # Given
C = np.array([1, 1, 1, 1]) # Starting values


# Chapter 4.3 in the book

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
    
    
r_1 = np.linspace(-5,5,100)
Wave = np.zeros(len(r_1))
for i in range(len(r_1)):
    Wave[i] = np.sum(C * Chi(r_1[i], alpha))

fig, ax = plt.subplots()

ax.plot(r_1,Wave)
plt.xlabel('r')
plt.ylabel('$\phi$(r)')

#print('Task 1:')
#print('Wave function:', Wave)
#print('')
print('Ground state energy of helium atom in a.u.:', E)



# Task 2

  
def x_stuff(a, b, n):
    h = (b-a)/n
    x_i = np.zeros((n))
    for i in range(n):
        x_i[i] = a + i*h
    return x_i, h
    

def density(r, alpha, C): # n_s(r) = 2*np.abs(phi(r))**2
    n = 2*np.abs(np.sum(C*Chi(r,alpha)))**2
    return n


def Hatree_pot(r):
    V = np.zeros(len(r))
    for i in range(len(r)):
        V[i] = 1/r[i] - (1+1/r[i])*np.exp(-2*r[i])
    return V

# Solve with finite differences
a = 0.0001
b = 10
n = 1000
h = x_stuff(a, b, n+1)[1]
rrr = x_stuff(a, b, n+1)[0]
A = np.zeros((n+1, n+1))
A[0, 0] = 1
A[n, n] = 1
#A[0, 1] = 1
#A[n, n-1] = 1
u_sqq = np.zeros((n+1))

for i in range(1, n):
    A[i, i-1] = 1
    A[i, i] = -2
    A[i, i+1] = 1
    u_sqq[i] = -h**2*2*np.pi*density(rrr[i], alpha,C)*rrr[i]
    
u_sqq[0] = 0
u_sqq[-1] = 1   
U = np.linalg.solve(A,u_sqq) # Solve: AU = u_sqq


#print(u_sqq)


#fig1, ax1 = plt.subplots()
#ax1.plot(rrr,U/rrr)
#ax1.plot(rrr,Hatree_pot(rrr),'r--')
#ax1.legend(['FDM', 'Hatree'])






# Task 3



def V_xc(r): # Page 3 in the assignment sheet
    A = 0.0311
    B = -0.048
    C = 0.002
    D = -0.0116
    gamma = -0.1423
    beta_1 = 1.0529
    beta_2 = 0.3334
    
    n = np.zeros(len(r))
    eps_x = np.zeros(len(r))
    eps_c = np.zeros(len(r))
    
    
    for i in range(len(r)):
        n[i] = 3/(4*np.pi*r[i]**3)
        #eps_x
        eps_x[i] = -3/4 * (3*n[i]/np.pi)**(1/3)
        
        # eps_c
        if r[i] >= 1:
            eps_c[i] = gamma/(1+beta_1*np.sqrt(r[i])+beta_2*r[i])
        else:
            eps_c[i] = A*np.log(r[i])+B+C*r[i]*np.log(r[i])+D*r[i]
            
        eps_xc = eps_x + eps_c
        
    
    return eps_xc - n**(1/3)/4*(r/np.pi)**(1/3)










