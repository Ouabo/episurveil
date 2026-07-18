"""Small-population Gillespie simulation for a minimal SVEAIHCRD system."""
import numpy as np
def transition_events(x,beta=.35,sigma=1/4.5,gamma_i=1/8):
    S,V,E,A,I,H,C,R,D=x; n=max(S+V+E+A+I+H+C+R,1.)
    return [(beta*S*(I+.5*A)/n,np.array([-1,0,1,0,0,0,0,0,0.])),(sigma*E*.62,np.array([0,0,-1,0,1,0,0,0,0.])),(sigma*E*.38,np.array([0,0,-1,1,0,0,0,0,0.])),(gamma_i*I,np.array([0,0,0,0,-1,0,0,1,0.]))]
def gillespie(x0,horizon,seed=0):
    rng=np.random.default_rng(seed); t=0.; x=np.asarray(x0,float).copy(); out=[(t,x.copy())]
    while t<horizon:
        events=transition_events(x); rates=np.array([max(r,0.) for r,_ in events]); total=rates.sum()
        if total<=0: break
        t+=rng.exponential(1/total); k=rng.choice(len(events),p=rates/total); x=np.maximum(x+events[k][1],0.); out.append((t,x.copy()))
    return out
