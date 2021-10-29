import numpy as np
from scipy import io as spio
from scipy import optimize
from numpy import vectorize, interp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
#from matplotlib.ticker import MaxNLocator

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
#  print(r, theta, phi, q, a, r1, R, np.array([a,0,0]) + np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)]), -(R**2*(1+q)/(2*a**3) + 1/r1 + q/r))
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

vgetL = np.vectorize(getL)
qlist = np.logspace(-5,0,50)
Lx,Ly,Lz = vgetL(qlist,1)

def funcargs(r, *args):
  func = lambda x,q:pot3D(x, np.pi/2, np.pi/2, q, 1)
  if (r>0):
    return -func(r[0], args[0])
  else:
    return -func(-r[0], args[0])

def minPy(q,Lx):
  return optimize.fmin(funcargs,Lx,args=(q,),disp=0)

vminPy = np.vectorize(minPy)

L4 = np.fabs(vminPy(qlist,Lx))

q_mesh = 1E-4
boxsize = 5*np.interp(q_mesh,qlist,Lx)
x = np.linspace(-boxsize,boxsize,100)
y = np.linspace(0,2*boxsize,100)
X,Y = np.meshgrid(x,y)
r_mesh = np.sqrt(X**2+Y**2)
phi_mesh = np.arccos(X/r_mesh)

vpot3D = np.vectorize(pot3D, excluded=['theta','q','a'])
Z = vpot3D(r_mesh,np.pi/2.,phi_mesh,q_mesh,1)
#levels = MaxNLocator(nbins=15).tick_values(Z.min(), Z.max())

# ry = np.linspace(q_mesh,2*boxsize,100)
# poty = vpot3D(ry,np.pi/2., np.pi/2., q_mesh,1)

#print(np.transpose([ry,poty]))

plt.plot(qlist, Lx)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('M2/M1',fontsize=16)
plt.ylabel('L1/a',fontsize=16)

plt.figure()
plt.plot(qlist, Ly/Lx, 'k-', label='y/x')
plt.plot(qlist, Lz/Lx, 'b-', label='z/x')
plt.plot(qlist, Lx/L4, 'r-', label='Lx/L4')
plt.xscale('log')
plt.xlabel('M2/M1',fontsize=16)
plt.ylabel('Ratio',fontsize=16)
plt.legend(loc=0, frameon = False)

plt.figure()
plt.plot(qlist, L4, 'k-')
plt.xscale('log')
plt.xlabel('M2/M1',fontsize=16)
plt.ylabel('distance to minimum P in y direction',fontsize=16)

plt.figure()
plt.contourf(X,Y,np.log(Z.max()-Z+1E-5))

# plt.figure()
# plt.plot(ry,poty)

#print(optimize.fmin(funcargs,np.interp(q_mesh,qlist,Lx),args=(q_mesh,),disp=1,full_output=1))
plt.show()
