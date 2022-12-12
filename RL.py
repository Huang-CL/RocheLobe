import numpy as np
import sys
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

G = 6.6743e-8                     # in cgs CODATA 2018
#G = 6.67384e-8                     # in cgs CODATA 2010
Msun = 1.98840987E33
RJ = 7.1492E9
Rsun=6.957E10
MJ = 1.8981246E30            # Jupiter mass defined by IAU
AU = 1.495978707E13

# # HD 189733b
# Rstar = 0.805*Rsun
# Mstar = 0.8*Msun
# Rpt = 1.216*RJ
# Mpl = 1.138*MJ	
# Per = 191685

# # WASP 121b original
# Rpt = 1.865*RJ
# Rstar = 1.458*Rsun  # Delrez et al. 2016, self-consistent with planet parameters, 1.59 * 6.96*10^10 cm from GAIA DR2
# Mstar = 1.353*Msun
# Mpl = 1.183*MJ
# Per = 110153
# Rtop = 246688720000                 # radius of the top grid in the hydrodynamic outflow.

# # WASP 121b Delrez actual using IAU RJ values
# Rpt = 1.766*RJ
# Rstar = 1.4572*Rsun  
# Mstar = 1.3521*Msun
# Mpl = 1.1824*MJ
# Per = 110153.56


# # WASP-52b
# Rstar = 0.79*Rsun
# Mstar = 0.87*Msun
# Rpt = 1.27*RJ
# Mpl = 0.46*MJ	
# Per = 155181

#WASP-12b Collins et al. 2017
Rstar = 1.657*Rsun;	
Mstar = 1.434*Msun;
Mpl = 1.47*MJ;	
Rpt = 1.90*RJ;
Per = 94299

# #WASP-12b Budaj 2011
# Rstar = 1.57*Rsun;	
# Mstar = 1.35*Msun;
# Mpl = 1.41*MJ;	
# Rpt = 1.7681*RJ;
# Per = 94299

# #WASP-33b Budaj 2011
# Rstar = 1.444*Rsun;	
# Mstar = 1.495*Msun;
# Mpl = 1.11*MJ;	
# Rpt = 1.5469*RJ;
# Per = 105397

a_ex = np.cbrt(G*(Mstar+Mpl)*(Per/(2*np.pi))**2)
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


def getequiryz(theta, rin, tin, yM, q, a):
  # Get the equipotential radius in the y-z plane that has the same potential as the rin, tin point at the direction of theta
  P0 = pot3D(rin,tin,np.pi/2,q,a)
  PyM = pot3D(yM,np.pi/2,np.pi/2,q,a)
  if rin>yM or P0>PyM:
    print("Requested reference point outside yMax.")
    return float("nan")
  return optimize.root_scalar(equiP, args=(theta,np.pi/2,q,a,P0), bracket=[yM/1E10, yM], method='brentq').root


def getL(q, a):
  L1x = optimize.root_scalar(grav, args=(q,a), bracket=[a*q/2., a/2.], method='brentq').root
  L1y = getequir(np.pi/2, np.pi/2, L1x, np.pi/2, np.pi, L1x, q, a)
  L1z = getequir(0, np.pi/2, L1x, np.pi/2, np.pi, L1x, q, a)
  return L1x,L1y,L1z

def funcargs(r, *args):
  func = lambda x,q:pot3D(x, np.pi/2, np.pi/2, q, 1)
  if (r>0):
    return -func(r[0], args[0])
  else:
    return -func(-r[0], args[0])

def minPy(q,Lx):
  return optimize.fmin(funcargs,Lx,args=(q,),disp=0)

def y2z(theta, rin, q, a, rmax):
  # convert a point on the y axis to the equal potential point at the y-z plane.
  if (rin>rmax):
    P0 = pot3D(rmax, np.pi/2, np.pi/2, q, a)
  else:
    P0 = pot3D(rin, np.pi/2, np.pi/2, q, a)

  req = optimize.root_scalar(equiP, args=(theta,np.pi/2,q,a,P0), bracket=[rmax/1E10, rmax], method='brentq').root
  
  if(rin>rmax):
    return req/rmax*rin;
  else:
    return req;

def Eggleton(q,a):
  return a*0.49*q**(2./3.)/(0.6*q**(2./3.)+np.log(1+q**(1./3.)))
  
Lx,Ly,Lz = getL(Mpl/Mstar,a_ex)

#print (G*Mstar*(pot3D(Lx, np.pi/2, np.pi, Mpl/Mstar, a_ex)-pot3D(1.93E+10,  np.pi/2, np.pi, Mpl/Mstar, a_ex)))

# vgetL = np.vectorize(getL)
# qlist = np.logspace(-5,0,50)
# Lx,Ly,Lz = vgetL(qlist,1)
 
# vminPy = np.vectorize(minPy)

# L4 = np.fabs(vminPy(qlist,Lx))
# vy2z = np.vectorize(y2z, excluded=['theta', 'a'])
# L4z = vy2z(0, L4, qlist, 1, L4)

# q_med = 8.865E-4
# Rpy_med = 0.03528
# vy2z_angle = np.vectorize(y2z, excluded=['rin', 'q', 'a', 'rmax']) # Given rin, q, a, plot the equal potential surfice in the y-z plane
# angle = np.linspace(0,np.pi/2)
# L4_121 = np.interp(q_med,qlist,L4)
# radius_pot = vy2z_angle(angle, Rpy_med, q_med, 1, L4_121)
# b_121 = y2z(0, Rpy_med, q_med, 1, L4_121)
# radius_elliptical = Rpy_med*b_121/np.sqrt((b_121*np.sin(angle))**2 + (Rpy_med*np.cos(angle))**2)


# q_mesh = 1E-4
# boxsize = 3*np.interp(q_mesh,qlist,Lx)
# x = np.linspace(-boxsize,boxsize,200)
# y = np.linspace(0,2*boxsize,200)
# z = np.linspace(0,0.5*boxsize,200)
# X,Y = np.meshgrid(x,y)
# r_mesh = np.sqrt(X**2+Y**2)
# phi_mesh = np.arctan2(Y,X)

# X2,Z2 = np.meshgrid(x,z)
# r2_mesh = np.sqrt(X2**2+Z2**2)
# phi2_mesh = np.arctan2(0,X2)
# th2_mesh = np.arctan2(X2,Z2)

# vpot3Dxy = np.vectorize(pot3D, excluded=['theta','q','a'])
# potxy = vpot3Dxy(r_mesh,np.pi/2.,phi_mesh,q_mesh,1)
# vpot3Dxz = np.vectorize(pot3D, excluded=['q','a'])
# potxz = vpot3Dxz(r2_mesh,th2_mesh,phi2_mesh,q_mesh,1)
#levels = MaxNLocator(nbins=15).tick_values(potxy.min(), potxy.max())

# ry = np.linspace(q_mesh,2*boxsize,100)
# poty = vpot3D(ry,np.pi/2., np.pi/2., q_mesh,1)

#print(np.transpose([ry,poty]))

# plt.plot(qlist, Lx)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('M2/M1',fontsize=16)
# plt.ylabel('L1/a',fontsize=16)

# plt.figure()
# plt.plot(qlist, Ly/Lx, 'k-', label='y/x')
# plt.plot(qlist, Lz/Lx, 'b-', label='z/x')
# plt.plot(qlist, Lx/L4, 'r-', label='Lx/L4')
# plt.plot(qlist, L4z/L4, 'm-', label='L4z/L4')
# plt.xscale('log')
# plt.xlabel('M2/M1',fontsize=16)
# plt.ylabel('Ratio',fontsize=16)
# plt.legend(loc=0, frameon = False)

# plt.figure()
# plt.plot(qlist, L4, 'k-')
# plt.xscale('log')
# plt.xlabel('M2/M1',fontsize=16)
# plt.ylabel('distance to minimum P in y direction',fontsize=16)

# my_dpi=96
# plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
# plt.contourf(X,Y,np.log(potxy.max()-potxy+1E-5),20)
# plt.xlabel('x/a',fontsize=16)
# plt.ylabel('y/a',fontsize=16)
# plt.tight_layout()

# plt.figure(figsize=(1600/my_dpi, 500/my_dpi), dpi=my_dpi)
# plt.contourf(X2,Z2,np.log(potxz.max()-potxz+1E-5),20)
# plt.xlabel('x/a',fontsize=16)
# plt.ylabel('z/a',fontsize=16)
# plt.tight_layout()

# # plt.figure()
# # plt.plot(ry,poty)

# #print(optimize.fmin(funcargs,np.interp(q_mesh,qlist,Lx),args=(q_mesh,),disp=1,full_output=1))

# #print(L4_121, b_121, b_121*1.495978707E13*0.02495/7.1492E9, np.sqrt(b_121*Rpy_med)*1.495978707E13*0.02495/7.1492E9)


# # Plot the equal potential radius at y-z plane and compare its shape to ellipse
# fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace' : 0})
# ax1.plot(angle,radius_pot,'b-',label='equal potential')
# ax1.plot(angle,radius_elliptical, 'k-', label='ellipse')
# ax2.plot(angle,radius_pot/radius_elliptical, 'k-')
# ax1.legend(loc=0, frameon = False)
# ax1.set_ylabel('r',fontsize=16)
# ax2.set_ylabel('potential/ellipse',fontsize=16)
# ax2.set_xlabel(r'$\theta$',fontsize=16)
# plt.tight_layout()

def RT(Rz):
  # Given the polar radius, calculate the difference between np.sqrt(Ry*Rz) and measured transit radius.
  return getequir(np.pi/2, np.pi/2, Rz, 0, np.pi, Lx, Mpl/Mstar, a_ex)*Rz - Rpt**2

Rz = optimize.root_scalar(RT, method='secant', x0=Rpt, x1=0.99*Rpt).root
Ry = getequir(np.pi/2, np.pi/2, Rz, 0, np.pi, Lx, Mpl/Mstar, a_ex)
Rx = getequir(np.pi/2, np.pi, Rz, 0, np.pi, Lx, Mpl/Mstar, a_ex)
yMax = minPy(Mpl/Mstar, Lx/a_ex)[0]
#print(yMax*a_ex/Rpt, Rstar/Rpt)
angle = np.linspace(0,2*np.pi, 129)
RL = [getequir(np.pi/2, i, Lx, np.pi/2, np.pi, Lx, Mpl/Mstar, a_ex) for i in angle]
Rplist = [getequir(np.pi/2, i, Ry, np.pi/2, np.pi/2, Lx, Mpl/Mstar, a_ex) for i in angle]
Rpyz = [getequir(i, np.pi/2, Ry, np.pi/2, np.pi/2, Lx, Mpl/Mstar, a_ex) for i in angle]
RhLyz = [getequiryz(i, yMax*a_ex/2, np.pi/2, yMax*a_ex, Mpl/Mstar, a_ex) for i in angle]
RLyz = [getequiryz(i, yMax*a_ex, np.pi/2, yMax*a_ex, Mpl/Mstar, a_ex) for i in angle]
RRLyz= [getequiryz(i, Ly, np.pi/2, yMax*a_ex, Mpl/Mstar, a_ex) for i in angle]
#Rstaryz = np.array(RLyz)*(Rstar/RLyz[0])
#Rtopyz = np.array(RLyz)*(Rtop*Ly/Lx/RLyz[25])

plt.figure()
plt.subplot(111, projection='polar')
plt.plot(angle, RL)
plt.plot(angle, Rplist)
plt.fill_between(angle, 0, Rplist, alpha=0.2)
plt.plot((0, np.pi/2), (0, Ry))
plt.plot((0, np.pi), (0, Rx))
plt.text(np.pi, Lx*0.85, "L1",fontsize=24)
plt.text(np.pi/2*0.9, Ry*0.5, "$R_{py}$",fontsize=24)
plt.text(np.pi*1.15, Rx*0.6, "$R_{px}$",fontsize=24)
plt.gca().set_rmin(0)
plt.gca().set_rmax(Lx)
plt.grid(False)
plt.axis('off')
plt.subplots_adjust(left=0, bottom=-0.1, right=1.05, top=1.1, wspace=0, hspace=0)

plt.figure()
plt.subplot(111, projection='polar')
plt.plot(angle+np.pi/2, Rpyz,'k')
plt.plot(angle+np.pi/2, RRLyz,'b')
plt.plot(angle+np.pi/2, RhLyz,'m')
plt.plot(angle+np.pi/2, RLyz,'r')
# plt.plot(angle+np.pi/2, Rtopyz,'m')
#plt.plot(angle+np.pi/2, Rstaryz,'r')
plt.plot(angle,np.ones(129)*Rstar, 'y')
plt.fill_between(angle+np.pi/2, 0, Rpyz, color='k',alpha=0.8)
plt.plot((0, np.pi/2), (0, Rz),color='w')
plt.plot((0, np.pi), (0, Ry),color='w')
#plt.text(np.pi, Lx*1.1, "L1")
plt.text(np.pi/2, Ry*1.1, "$R_{pz}$",fontsize=30)
plt.text(np.pi, Ry*2.7, "$R_{py}$",fontsize=30)
plt.gca().set_rmin(0)
#plt.gca().set_rmax(Rstaryz[25]*1.05)
plt.gca().set_rmax(Rstar*1.02)
plt.grid(False)
plt.axis('off')
#plt.subplots_adjust(left=0, bottom=-0.6, right=1, top=1.6, wspace=0, hspace=0)
plt.subplots_adjust(left=0, bottom=-0.3, right=1, top=1.3, wspace=0, hspace=0)

RLE = Eggleton(Mpl/Mstar,a_ex)
print(Rx/RJ, Ry/RJ, Rz/RJ, Rx/Rz, a_ex/AU, 2*np.pi*np.sqrt((0.02544*AU)**3/G/(Mpl+Mstar)), 2*np.pi*Ry/Per)
print (angle[16]/np.pi*180,Lx/RJ, Ly/RJ, Lz/RJ, Lx/Rpt, Ly/Rstar, RLE/RJ, RRLyz[16]/RJ, RRLyz[16]/Ly, RRLyz[16]/Rstar, RRLyz[16]/Rx, RRLyz[16]/np.sqrt(Ly*Lz))

plt.show()
