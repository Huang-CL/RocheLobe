import numpy as np
from scipy import io as spio
from scipy import optimize
from numpy import vectorize
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

rc('text',usetex=True)          
mpl.rcParams['font.size']=16.5
mpl.rcParams["savefig.directory"] = ""
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)

# M1 = 1, M2 = q
# Mass in unit of MSun, a in unit of AU


def grav(r, *args):
  # Gravity in co-rotating frame at r from planet
  q = args[0]
  a = args[1]
  return q/np.square(r) + 1/np.square(a) - (1+q)*r/a**3 - 1/np.square(a-r)

def pot(r, *args):
  # g(r) dr along the line of two body. gravitational potential + constant
  q = args[0]
  a = args[1]
  return 1/a**2*r - q/r - (1+q)*r**2/(2*a**3) - 1/(a-r)

def pot3D(r, theta, phi, q, a):
# three dimensional potential.  Substellar point theta = pi/2, phi = pi
  r1 = np.linalg.norm(np.array([a,0,0]) + np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)]))
  R = np.linalg.norm(np.array([a*1/(1+q),0]) + np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi)]))
  return -(R**2*(1+q)/(2*a**3) + 1/r1 + q/r)

def equiP(r, *args):
  return pot3D(r,args[0],args[1],args[2],args[3])-args[4]

def getequir(theta, phi, rin, tin, pin, L1, q, a):
  # Get the equipotential radius within the secondary Roche Lobe that has the same potential as the rin, tin, pin point at the direction of theta, phi
  P0 = pot3D(rin,tin,pin,q,a)
  PL1 = pot3D(L1,np.pi/2,np.pi,q,a)
  if rin>L1 or P0>PL1:
    print("Requested reference point outside secondary Roche Lobe.")
    return float("nan")
  return optimize.root_scalar(equiP, args=(theta,phi,q,a,P0), bracket=[L1/1E10, L1], method='brentq').root

def getL(q, a):
  L1x = optimize.root_scalar(grav, args=(q,a), bracket=[a*q/2., a/2.], method='brentq').root
  L1y = getequir(np.pi/2, np.pi/2, L1x, np.pi/2, np.pi, L1x, q, a)
  L1z = getequir(0, np.pi/2, L1x, np.pi/2, np.pi, L1x, q, a)
  return L1x,L1y,L1z

qlist = np.logspace(-5,0,50)
Lx = np.zeros(50)
Ly = np.zeros(50)
Lz = np.zeros(50)

for i in range(len(qlist)):
  Lx[i],Ly[i],Lz[i]=getL(qlist[i],1)

plt.plot(qlist, Lx)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('M2/M1',fontsize=16)
plt.ylabel('L1/a',fontsize=16)

plt.figure()
plt.plot(qlist, Ly/Lx, 'k-', label='y/x')
plt.plot(qlist, Lz/Lx, 'b-', label='z/x')
#plt.plot(qlist, Lz/Ly, 'r-', label='z/y')
plt.xscale('log')
plt.xlabel('M2/M1',fontsize=16)
plt.ylabel('Ratio',fontsize=16)
plt.legend(loc=0, frameon = False)

plt.show()