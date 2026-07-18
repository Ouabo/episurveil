"""Deterministic Euler simulation for coupled SVEAIHCRD patches."""
import numpy as np
from .sveaihcrd import Parameters
def rhs_multipatch(x,beta,mobility,p=None):
    p=p or Parameters(); x=np.asarray(x,float); P=x.shape[0]; out=np.zeros_like(x)
    living=np.maximum(x[:,:8].sum(axis=1),1e-12); infectious=(x[:,4]+p.eta_a*x[:,3])/living
    lam=np.asarray(beta)*(np.asarray(mobility)@infectious)
    for i in range(P):
        S,V,E,A,I,H,C,R,D=x[i]; lv=lam[i]
        out[i]=[-lv*S-p.nu*S+p.omega_v*V+p.omega_r*R,p.nu*S-(1-p.vaccine_efficacy)*lv*V-p.omega_v*V,lv*S+(1-p.vaccine_efficacy)*lv*V-p.sigma*E,(1-p.kappa)*p.sigma*E-p.gamma_a*A,p.kappa*p.sigma*E-(p.gamma_i+p.tau_i)*I,p.tau_i*I-(p.gamma_h+p.tau_h+p.delta_h)*H,p.tau_h*H-(p.gamma_c+p.delta_c)*C,p.gamma_a*A+p.gamma_i*I+p.gamma_h*H+p.gamma_c*C-p.omega_r*R,p.delta_h*H+p.delta_c*C]
    return out
def simulate_multipatch(x0,times,beta,mobility,p=None):
    x=np.zeros((len(times),)+np.asarray(x0).shape); x[0]=x0
    for k in range(1,len(times)):
        dt=times[k]-times[k-1]; x[k]=np.maximum(x[k-1]+dt*rhs_multipatch(x[k-1],beta,mobility,p),0.)
        x[k,:,8]=np.maximum(x[k,:,8],x[k-1,:,8])
    return x
