import FVis2 as FVis
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage.interpolation import shift
import sys

"""
Pick up the parameters in the Parameter.py file. Contains all the solar parameters
"""

if len(sys.argv)<2:
    print 'Wrong number of input arguments.'
    print 'Usage: python Convection.py parameter.py'
    sys.exit()
namespace={}
paramfile = sys.argv[1]
execfile(paramfile,namespace)
globals().update(namespace)


class HydroDynamics():

    def __init__(self, N):

        self.N = N                          # time steps
        self.ny = 100                       # 4 Mm
        self.nx = int(3*self.ny)            # 12 Mm

        # Inital Solar parmeters:
        self.rho0 = rhoPhotos                # density in the Photoshere
        self.T0 = TempPhotos                 # Temperature in the Photoshere
        self.P0 = PressurePhotos             # Pressure in the Photoshere

        # Constants
        self.mu = 0.61
        self.m_u = 1.660539e-27                             # kg
        self.m_p, self.m_e = 1.6726e-27, 9.1094e-31         # kg,kg
        self.e, self.k_B = 1.602e-19, 1.381e-23             # C, m^2kgs^-2K^-1
        self.eps0 = 8.954e-12                               # Fm^-1
        self.h = 6.627e-34; self.avo = 6.0221e+23           # Js, mol^-1
        self.c = 299792458.                                 # m/s
        self.sigma = 5.67e-8                                # Wm^-2 K^-4
        self.a = 4*self.sigma/float(self.c)
        self.G = 6.672e-11                                  # Nm^2 kg^-2
        self.F = self.G*SolarMass/(SolarRadius**2)          # Gravity, m/s^2
        self.Del = 2/5.*1.1
        self.gamma = 5./3.

        # Arrays
        self.x, self.dx = np.linspace(0.0, 12e+6, self.nx, retstep=True)
        self.y = np.linspace(4e+6, 10, self.ny)

        self.T = np.zeros((self.nx,self.ny,self.N))
        self.P = np.zeros((self.nx,self.ny,self.N))
        self.rho = np.zeros((self.nx,self.ny,self.N))
        self.e = np.zeros((self.nx,self.ny,self.N))
        self.ux = np.zeros((self.nx,self.ny,self.N))
        self.uy = np.zeros((self.nx,self.ny,self.N))

        # Set initial and boundary conditions for the different arrays
        self.P[:,0,0] = self.P0
        self.T[:,0,0] = self.T0
        self.rho[:,0,0] = self.rho0
        self.ux[:,:,0] = 0.0; self.uy[:,:,0] = 0.0
        self.ux[:,0,:] = 0; self.ux[:,-1,:] = 0; self.ux[0,:,:] = 0; self.ux[-1,:,:] = 0
        self.uy[:,0,:] = 0; self.uy[:,-1,:] = 0; self.uy[0,:,:] = 0; self.uy[-1,:,:] = 0


    def Gradient(self, f):

        f[0,:] = f[-1,:]            # BC for x direction
        # in x direction:
        ddx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))/(2*self.dx)

        # In y direction:
        ddy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))/(2*self.dx)
        # BC for y direction
        ddy[:,0] = (ddy[:,0] - ddy[:,1])/self.dx
        ddy[:,-1] = (ddy[:,-1] - ddy[:,-2])/self.dx

        grad = [ddx, ddy]
        return grad

    def drhodt(self, rho, ux, uy):

        drhodt = -self.Gradient(rho*ux)[0] - self.Gradient(rho*uy)[1]
        return drhodt

    def duxdt(self, ux,uy,rho,P):

        term1 = -self.Gradient(P)[0]/rho
        term2 = ux*(self.Gradient(ux)[0] + self.Gradient(uy)[1])
        return term1 - term2

    def duydt(self, ux,uy,rho,P):

        term1 = self.F - self.Gradient(P)[1]/rho
        term2 = uy*(self.Gradient(ux)[0] + self.Gradient(uy)[1])
        return term1 - term2


    def dedt(self,ux,uy,e,P):

        dedt = -self.Gradient(ux*e)[0] - self.Gradient(uy*e)[1] - P*(self.Gradient(ux)[0] + self.Gradient(uy)[1])
        return dedt


    def dPdy(self, rho):

        return self.F*rho


    def dTdy(self,T,P,rho):
        """
        Compute the derivative of temperature
        """

        dTdy = self.Del*(T/P)*self.dPdy(rho)
        return dTdy

    def Density(self, T, P):

        return self.mu*self.m_u*P/(self.k_B*T)



    def dt(self, rel_rho, rel_ux, rel_uy, rel_e):

        p = 0.1
        dt1 = np.max(rel_rho)
        dt2 = np.max(rel_e)
        dt3 = np.max(rel_ux)
        dt4 = np.max(rel_uy)

        dt = [dt1,dt2,dt3,dt4]
        if all(dt) == 0.0:
            dt = p
        else:
            d = max([x for x in dt if x != 0])
            dt = p/d

        return dt


    def Integrate(self):

        rho = self.rho; e = self.e; T = self.T; P = self.P; ux = self.ux; uy = self.uy
        drhodt = self.drhodt; dedt = self.dedt; duxdt = self.duxdt; duydt = self.duydt
        dx = self.dx
        # Set up the initial box:
        rho[:,0,0] = self.Density(self.P0, self.T0)
        P[:,0,0] = self.P0; T[:,0,0] = self.T0

        for j in range(self.ny-1):
            P[:,j+1,0] = P[:,j,0] + dx*self.dPdy(rho[:,j,0])
            T[:,j+1,0] = T[:,j,0] + dx*self.dTdy(T[:,j,0], P[:,j,0],rho[:,j,0])
            rho[:,j+1,0] = self.Density(P[:,j+1,0], T[:,j+1,0])

        e[:,:,0] = (self.gamma - 1)*rho[:,:,0]*self.k_B*T[:,:,0]/(self.mu*self.m_u)
        print P[100,:,0]
        sys.exit()
        # BC for temperature
        T[:,0,0] = 0.9*T[:,0,0]
        T[:,-1,0] = 1.1*T[:,-1,0]

        delta = 10;         dtlist = []
        # Integrate over time:
        for n in range(self.N-1):

            ux[0,:,n] = 0.0
            uy[:,0,n] = 0.0
            ux[-1,:,n] = 0.0
            uy[:,-1,n] = 0.0

            # Time step computing:
            rel_rho = abs(drhodt(rho[:,:,n], ux[:,:,n], uy[:,:,n])/rho[:,:,n])
            rel_e = abs(dedt(ux[:,:,n], uy[:,:,n], e[:,:,n], P[:,:,n])/e[:,:,n])
            rel_ux = abs(self.ux[:,:,n]/dx)
            rel_uy = abs(self.uy[:,:,n]/dx)
            dt = self.dt(rel_rho,rel_e,rel_ux,rel_uy)

            dtlist.append(dt)

            # integration
            drho = dt * self.drhodt(rho[:,:,n], ux[:,:,n], uy[:,:,n])
            dux = dt * self.duxdt(ux[:,:,n], uy[:,:,n], rho[:,:,n], P[:,:,n])
            duy = dt * self.duydt(ux[:,:,n], uy[:,:,n], rho[:,:,n], P[:,:,n])
            de = dt * self.dedt(ux[:,:,n], uy[:,:,n], e[:,:,n], P[:,:,n])

            #Boundary conditions for energy
            elower = e[:,-2,n]*0.9
            eupper = e[:,0,n]*1.1
            if any(abs(ux[:,-1,n])) < 1.0:
                de[:,-1] = (e[:,-2,n] - elower)/delta

            if any(abs(ux[:,0,n])) < 1.0:
                de[:,0] = (e[:,1,n] - eupper)/delta

            rho[:,:,n+1] = drho + 0.25*dx*sum(self.Gradient(rho[:,:,n]))
            ux[:,:,n+1] = dux + 0.25*dx*sum(self.Gradient(ux[:,:,n]))
            uy[:,:,n+1] = duy + 0.25*dx*sum(self.Gradient(uy[:,:,n]))
            e[:,:,n+1] = de + 0.25*dx*sum(self.Gradient(e[:,:,n]))

            P[:,:,n+1] = e[:,:,n+1]/(self.gamma - 1)
            T[:,:,n+1] = self.mu*self.m_u*P[:,:,n+1]/(self.k_B*rho[:,:,n+1])


        return T

    def step(self):

        # calculate the derivatives
        dPdy = self.dPdy(self.rho)
        dTdy = self.dTdy(self.T,self.P,self.rho)
        drhodt = self.drhodt(self.rho, self.ux, self.uy)
        dedt = self.dedt(self.ux, self.uy, self.e, self.P)
        duxdt = self.duxdt(self.ux, self.uy,self.rho, self.P)
        duydt = self.duydt(self.ux, self.uy,self.rho, self.P)

        # Find dt
        rel_rho = abs(drhodt(self.rho, self.ux, self.uy)/self.rho)
        rel_e = abs(dedt(self.ux, self.uy, self.e, self.P)/self.e)
        rel_ux = abs(self.ux/self.dx)
        rel_uy = abs(self.uy/self.dx)
        dt = self.dt(rel_rho,rel_e,rel_ux,rel_uy)

        # Boundary conditions
        self.ux[0,:,:] = 0.0
        self.uy[:,0,:] = 0.0
        self.ux[-1,:,:] = 0.0
        self.uy[:,-1,:] = 0.0

        T[:,0,0] = 0.9*T[:,0,0]
        T[:,-1,0] = 1.1*T[:,-1,0]

        elower = e[:,-2,:]*0.9
        eupper = e[:,0,:]*1.1
        if any(abs(ux[:,-1,:])) < 1.0:
            de[:,-1] = (e[:,-2,:] - elower)/delta

        if any(abs(ux[:,0,:])) < 1.0:
            de[:,0] = (e[:,1,:] - eupper)/delta


        # Update time for the variables
        self.rho[:] = self.rho
        self.ux[:] = self.ux
        self.uy[:] = self.uy
        self.e[:] = self.e
        self.P[:] = self.P
        self.T[:] = self.T

        return dt


    def test(self):
        Y, X = np.meshgrid(self.y,self.x)
        f = np.sin(X*1e-6) + np.sin(Y*1e-6)


        #grad = (shift(f[2],-1) - shift(f[2],1))/(2*dx)
        #grad = (np.roll(f,-1) - np.roll(f,1))/(2*dx)
        #print np.shape(grad)
        grad = self.Gradient(f)[1]
        #plt.plot(x, f[0,0], '-r')
        plt.plot(self.x, grad, '-b')
        #plt.plot(x,np.cos(x), '-g')
        #plt.show()
        return grad

N = 100
HD = HydroDynamics(N)
test = HD.test()
inte = HD.Integrate()
