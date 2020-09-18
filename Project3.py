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
    print 'Usage: python Prosject1.py parameter.py'
    sys.exit()
namespace={}
paramfile = sys.argv[1]
execfile(paramfile,namespace)
globals().update(namespace)


class HydroDynamics():

    def __init__(self, N):
        """
        Send in the number of time step wanted as N.
        """
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
        self.F = -self.G*SolarMass/(SolarRadius**2)          # Gravity, m/s^2
        self.Del = 2/5.*1.1
        self.gamma = 5./3.
        self.delta = 1.0e-7

        # Arrays
        self.x, self.dx = np.linspace(0.0, 12e+6, self.nx, retstep=True)
        self.y = np.linspace(4e+6, 0, self.ny, endpoint=False)
        self.Fc = np.zeros(self.ny)
        self.dy = -self.dx

        self.P = np.zeros((self.nx,self.ny,self.N))         # Pressure array
        self.T = np.zeros((self.nx,self.ny,self.N))         # Temperature array
        self.rho = np.zeros((self.nx,self.ny,self.N))       # Mass/density array
        self.px = np.zeros((self.nx,self.ny,self.N))        # Momentum x array
        self.py = np.zeros((self.nx,self.ny,self.N))        # Momentum y array
        self.e = np.zeros((self.nx,self.ny,self.N))         # Energy array
        self.vx = np.zeros((self.nx,self.ny,self.N))        # Velocity in x
        self.vy = np.zeros((self.nx,self.ny,self.N))        # velocity in y

        self.P1 = np.zeros((self.nx,self.ny))         # Pressure array
        self.T1 = np.zeros((self.nx,self.ny))         # Temperature array
        self.rho1 = np.zeros((self.nx,self.ny))       # Mass/density array
        self.E = np.zeros((self.nx,self.ny))         # Energy array
        self.ux = np.zeros((self.nx,self.ny))        # Velocity in x
        self.uy = np.zeros((self.nx,self.ny))        # velocity in y
        self.P1[:,0] = self.P0
        self.T1[:,0] = self.T0
        self.rho1[:,0] = self.rho0
        # Set initial and boundary conditions for the different arrays
        self.P[:,0,0] = self.P0
        self.T[:,0,0] = self.T0
        self.rho[:,0,0] = self.rho0
        self.vx[:,:,0] = 0.0; self.vy[:,:,0] = 0.0
        self.vx[:,0,:] = 0; self.vx[:,-1,:] = 0; self.vx[0,:,:] = 0; self.vx[-1,:,:] = 0
        self.vy[:,0,:] = 0; self.vy[:,-1,:] = 0; self.vy[0,:,:] = 0; self.vy[-1,:,:] = 0




    def drhodt(self, rho, ux,uy):
        """
        Compute the Continuity equation to find the mass.
        """

        px = rho*ux; py = rho*uy
        drhodt = -self.Gradient(px)[1] - self.Gradient(py)[0]
        return drhodt


    def dpdt(self, p, vx,vy, rho, P):
        """
        Equation of motion, Calculate how the gas is moving for momentum in direction i.
        """

        term2 = p*(sum(self.Gradient(vx)) + sum(self.Gradient(vy)))
        term3 = (vx + vy)*self.Gradient(p)[1]
        term4 = (vx + vy)*self.Gradient(p)[0] + sum(self.Gradient(P))
        dpdt =  -term2 - term3 - term4
        return dpdt


    def dedt(self, vx, vy, E, P):
        """
        How the energy move in the star.
        """

        term1 = self.Gradient(vx*E)[1] + self.Gradient(vy*E)[0]
        term2 = P*(self.Gradient(vx)[1] + self.Gradient(vy)[0])
        dedt = -(term1 + term2)
        return dedt


    def dvxdt(self, vx,vy, rho, P):
        """
        Differential of velocity in x direction.
        """

        a = -self.Gradient(P)[1]/rho
        b = -vx * (self.Gradient(vx)[1] + self.Gradient(vy)[0])
        dvxdt = a + b
        return dvxdt


    def dvydt(self,vx,vy,rho,P):
        """
        Differential of velocity in y direction.
        """

        a = -self.Gradient(P)[0]/rho
        b = -vy * (self.Gradient(vy)[0] + self.Gradient(vx)[1]) + self.F
        dvydt = a + b
        return dvydt


    def Gradient(self, f):
        """
        Compute the derivative of the called function/ variable with respect to
        x and y. Return an array with the derivatives.
        """

        f[0,:] = f[-1,:]            # BC for x direction
        # In x direction
        ddx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))/(2*self.dx)
        ddx[0,:] = ddx[-1,:]        # BC for x direction

        # In y direction:
        ddy = -(np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))/(2*self.dy)
        # BC for y direction
        ddy[:, 0] = (4*f[:, 1]-f[:, 2]-3*f[:, 0])/(2*self.dy) # (f[:,0] - f[:,1])/self.dy
        ddy[:,-1] = (4*f[:,-2]-f[:,-3]-3*f[:,-1])/(2*self.dy) # (f[:,-1] - f[:,-2])/self.dy #

        # Derivate over the diagonal neighbour cells
        ddxy = (np.roll(np.roll(ddx,-1,axis=1),-1,axis=0) - np.roll(np.roll(ddx, 1,axis=1), 1,axis=0))/(np.sqrt(2.))
        ddyx = (np.roll(np.roll(ddy, 1,axis=0), 1,axis=1) - np.roll(np.roll(ddy,-1,axis=0),-1,axis=1))/(np.sqrt(2.))
        # Diagonal BC
        ddxy[:, 0] = (np.roll(ddx[:, 1],1,axis=0) + 2*ddx[:, 0] - np.roll(ddx[:, 0],-1,axis=0))/(np.sqrt(2))
        ddxy[:,-1] = (np.roll(ddx[:,-2],1,axis=0) + 2*ddx[:,-1] - np.roll(ddx[:,-1],-1,axis=0))/(np.sqrt(2))
        ddxy[0,0] = ddyx[-1,0]; ddxy[0,-1] = ddyx[-1,-1]

        grad = [ddx + ddxy, ddy + ddyx]
        return grad


    def dPdy(self, rho):
        """
        compute the derivative of pressure from hydrostatics.
        """

        return self.F*rho


    def dTdy(self,T,P,rho):
        """
        Compute the derivative of temperature
        """

        dTdy = self.Del*(T/P)*self.dPdy(rho)
        return dTdy


    def Density(self, P, T):
        """
        Compute the derivative of the density
        """

        return (P/T)*self.mu*self.m_u/(self.k_B)


    def dt(self,vx,vy,e,rho,P):
        """
        Compute the time step from the variation of the variables.
        """

        p = 0.1
        rel_rho = np.max(abs(self.drhodt(rho,vx,vy)/rho))
        rel_e = np.max(abs(self.dedt(vx,vy,e,P)/e))
        rel_vx = np.max(abs(vx/self.dx))
        rel_vy = np.max(abs(vy/self.dy))
        dt1 = [rel_rho, rel_e, rel_vx, rel_vy]

        if all(dt1) == 0.0:
            dt = 0.1
        else:
            d = max([x for x in dt1 if x != 0])
            dt = p/d
            if dt > 10:
                dt = 1

        return dt


    def Set_Initial(self):
        """
        Set the initial conditions for the box using forward Euler method.
        """

        self.rho1[:,0] = self.Density(self.P0, self.T0)
        for j in range(self.ny-1):
            self.P1[:,j+1] = self.P1[:,j] + self.dy*self.dPdy(self.rho1[:,j])
            self.T1[:,j+1] = self.T1[:,j] + self.dy*self.dTdy(self.T1[:,j], self.P1[:,j],self.rho1[:,j])
            self.rho1[:,j+1] = self.Density(self.P1[:,j+1], self.T1[:,j+1])

        self.E[:,:] = (self.gamma - 1)*self.rho1[:,:]*self.k_B*self.T1[:,:]/(self.mu*self.m_u)

        if self.rho1.any() == 0:
            print 'Some values = 0'


    def Integrate(self):
        """
        integrate the variables over time
        """
        dx = self.dx
        dy = self.dy
        rho = self.rho; P = self.P; T = self.T
        px = self.px; py = self.py; e = self.e
        vx = self.vx; vy = self.vy
        drhodt = self.drhodt; dpdt = self.dpdt; dedt = self.dedt
        dvxdt = self.dvxdt; dvydt = self.dvydt

        # Inital conditions, filling the grids
        rho[:,0,0] = self.Density(self.P0, self.T0)

        for j in range(self.ny-1):
            P[:,j+1,0] = P[:,j,0] + dy*self.dPdy(rho[:,j,0])
            T[:,j+1,0] = T[:,j,0] + dy*self.dTdy(T[:,j,0], P[:,j,0],rho[:,j,0])
            rho[:,j+1,0] = self.Density(P[:,j+1,0], T[:,j+1,0])

        px[:,:,0] = rho[:,:,0]*self.vx[:,:,0]
        py[:,:,0] = rho[:,:,0]*self.vy[:,:,0]
        e[:,:,0] = (self.gamma - 1)*rho[:,:,0]*self.k_B*T[:,:,0]/(self.mu*self.m_u)

        # Boundry conditions:
        T[:,0,0] = 0.9*T[:,0,0]
        T[:,-1,0] = 1.1*T[:,-1,0]

        # Integration over time
        dtlist = []; T_up = []; T_down = []
        for n in range(self.N-1):

            # BC for velocities:
            vx[0,:,n] = 0.0
            vy[:,0,n] = 0.0
            vx[-1,:,n] = 0.0
            vy[:,-1,n] = 0.0


            # Integration
            drho = drhodt(rho[:,:,n],vx[:,:,n],vy[:,:,n]) #* dt
            dvx = dvxdt(vx[:,:,n],vy[:,:,n],rho[:,:,n],P[:,:,n]) #* dt
            dvy = dvydt(vx[:,:,n],vy[:,:,n],rho[:,:,n],P[:,:,n]) #* dt
            de = dedt(vx[:,:,n],vy[:,:,n],e[:,:,n],P[:,:,n]) #* dt

            # Time step computing:
            dt = self.dt(vx[:,:,n],vy[:,:,n],e[:,:,n],rho[:,:,n], P[:,:,n])
            dtlist.append(dt)

            #Boundary conditions
            T[:,0,n] = 0.9*T[:,0,n]
            T[:,-1,n] = 1.1*T[:,-1,n]

            elower = e[:,-2,n]*0.9
            eupper = e[:,0,n]*1.1
            if any(abs(vx[:,-1,n])) < 1.0:
                de[:,-1] = (e[:,-2,n] - elower)/(self.delta*dt)

            if any(abs(vx[:,0,n])) < 1.0:
                de[:,0] = (e[:,1,n] - eupper)/(self.delta*dt)

            rho[:,0,n] = (e[:,0,n]/(self.k_B*T[:,0,n]))*self.mu*self.m_u/(self.gamma-1)
            rho[:,-1,n] = (e[:,-1,n]/(self.k_B*T[:,-1,n]))*self.mu*self.m_u/(self.gamma-1)

            rho[:,:,n+1] = drho*dt + dx*(self.Gradient(rho[:,:,n])[1] + self.Gradient(rho[:,:,n])[0])/8.
            vx[:,:,n+1] = dvx*dt + dx*(self.Gradient(vx[:,:,n])[1] + self.Gradient(vx[:,:,n])[0])/8.
            vy[:,:,n+1] = dvy*dt + dx*(self.Gradient(vy[:,:,n])[1] + self.Gradient(vy[:,:,n])[0])/8.
            e[:,:,n+1] = de*dt + dx*(self.Gradient(e[:,:,n])[1] + self.Gradient(e[:,:,n])[0])/8.

            P[:,:,n+1] = e[:,:,n+1]/(self.gamma - 1)
            T[:,:,n+1] = self.mu*self.m_u*P[:,:,n+1]/(self.k_B*rho[:,:,n+1])


            # Print statements:
            if n%1 == 0:
                print 'i=%2d, dt=%.3e >>>'%(n,dt), np.max(rho[:,:,n]),np.mean(vx[:,:,n]),np.mean(vy[:,:,n]),np.max(e[:,:,n])

        print np.max(e[:,:,75]),np.max(P[:,:,75]),np.max(rho[:,:,75])

        self.rho = rho; self.vx = vx; self.vy = vy; self.e = e; self.T = T; self.P = P
        print '------', sum(dtlist)
        return dtlist


    def ConvectiveVelocity(self):
        """
        Compute the convective velocity and compare it to the vertical velocity
        of the gass.
        """

        T = self.T[:,:,75]; P = self.P[:,:,75]; rho = self.rho[:,:,75]
        uy = self.vy[:,:,75]; ux = self.vx[:,:,75]
        d = 1.0         # delta
        self.vel_conv = np.nan_to_num(np.sqrt(self.F*d*self.dTdy(P,T,rho)/T)*self.dx)

        xx,yy = np.meshgrid(self.y, self.x)
        fig = plt.figure('conv vel')
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, rho, cmap=cm.plasma)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m/s]')
        plt.tight_layout()
        plt.savefig('Density.png')

        print '------'
        print 'The convective velocity is,'
        print self.vel_conv
        print '------'
        print 'Difference between convective velocity and vertical velocity'
        print self.vel_conv - uy

        """
        Mass fraction moving with convective velocity +/- 10%. For each cell, the
        mass moving up with the given velocity range needs to be summed up. This
        gives the mass fraction moving with the given velocity range.
        """

        mass_y = []; mass_x = []
        for i in range(self.nx):
            for j in range(self.ny):
                if uy[i,j] >= self.vel_conv[i,j]*0.9 and uy[i,j] <= self.vel_conv[i,j]*1.1:
                    mass_y.append(rho[i,j])
                if ux[i,j] >= self.vel_conv[i,j]*0.9 and ux[i,j] <= self.vel_conv[i,j]*1.1:
                    mass_x.append(rho[i,j])

        MassFraction_y = np.sum(mass_y)/np.sum(rho)
        MassFraction_x = np.sum(mass_x)/np.sum(rho)
        print 'Fraction of mass moving with velocities v_conv +/- 10% in x direction:',MassFraction_x
        print 'Fraction of mass moving with velocities v_conv +/- 10% in y direction:',MassFraction_y

        print '-----------'
        return self.vel_conv

    def Del_star(self):
        """
        calculate the temperature gradient of the star.
        """

        P = self.P[:,:,75]; rho = self.rho[:,:,75]; T = self.T[:,:,75]
        Hp = -P/self.dPdy(rho)          # Pressure height scale
        self.Del_star = -Hp/T * self.dTdy(P,T,rho)
        print 'Del_star for the gass:'
        print self.Del_star

        """
        Find (Del_star - Del_p)
        """
        self.ksi2 = -self.Del_star
        print '(Del_star - Del_p) become:'
        print self.ksi2

        print '-------------'
        return self.Del_star


    def ConvectiveFlux(self):
        """
        Find the convective flux for every value of y.
        """

        e = self.e[:,:,75]; uy = self.vy[:,:,75]; ux = self.vx[:,:,75]
        Fc_up = 0; Fc_down = 0; e_list = []; uy_up = []; uy_down = []

        # Find index where y = 2e+6
        for j,val in enumerate(self.y):
            if val == 2e+6:
                ind = j

        self.Fc = np.sum(e*uy*self.dx, axis=0)
        Fc_horizontal = np.sum(e*ux*self.dx, axis=1)
        print 'Convective flux'
        print self.Fc

        # Plot of convection in y and x direction
        plt.figure('Fc vertical')
        plt.plot(self.y, self.Fc, '-b',label=r'$F_c(y)$')
        plt.legend(loc=1)
        plt.xlim(0,4.1e+6)
        plt.xlabel('y, [m]'); plt.ylabel(r'$F_c(y)$   $[Jm^2 s^{-1}]$')
        plt.savefig('ConvectiveFlux.png')

        plt.figure('Fc horozontal')
        plt.plot(self.x, Fc_horizontal, '-b',label=r'$F_c(x)$')
        plt.legend(loc=2)
        plt.xlim(0,12e+6)
        plt.xlabel('x, [m]'); plt.ylabel(r'$F_c(x)$   $[Jm^2 s^{-1}]$')
        plt.savefig('ConvFluxHorizontal.png')

        # Find convective flux going up and down
        for i in range(self.nx):
            if uy[i,ind] >= 0:
                Fc_up += (e[i,ind]*uy[i,ind]*self.dx)

            if uy[i,ind] < 0:
                Fc_down += (e[i,ind]*uy[i,ind]*self.dx)

        print 'Convective flux moving up at height 2 Mm is, Fc = %g'%(Fc_up)
        print 'Convective flux moving down at height 2 Mm is, Fc = %g'%(Fc_down)

        # Total energy flux from convection out of box:
        Fc_out = self.Fc[0]
        print Fc_out, np.sum(self.Fc)

        """
        Find the mixing length
        """
        P = self.P[:,:,75]; T = self.T[:,:,75]; rho = self.rho[:,:,75]
        d = 1.0
        Hp = -P/self.dPdy(rho)
        Cp = 2.5*self.k_B/(self.mu*self.m_u)
        a = np.nan_to_num(np.sqrt(T/(self.F*d*self.dTdy(P,T,rho))))
        lm = 2*Hp*self.Fc*a/(rho*Cp*self.dy*T*(self.ksi2))
        print 'Mixing length'
        print lm
        print np.max(lm), np.min(lm)

        # Find temperature difference
        DeltaT = 0.5*lm*self.dTdy(P,T,rho)
        print 'The temperature difference'
        print DeltaT
        print np.max(DeltaT)

        return self.Fc


    def step(self):
        """
        What to send in to the Fvis2 program to simulate in 2D.
        """

        # calculate the derivatives
        dPdy = self.dPdy(self.rho1)
        dTdy = self.dTdy(self.T1,self.P1,self.rho1)

        drhodt = self.drhodt(self.rho1,self.ux,self.uy)
        dedt = self.dedt(self.ux, self.uy, self.E, self.P1)
        elower = self.E[:,-1]*0.9
        eupper = self.E[:,0]*1.1
        #if (abs(self.ux[:,-1])).any() < 1.0:
        dedt[:,-1] = (self.E[:,-2] - elower)/(1*self.delta)
        #if (abs(self.ux[:,0])).any() < 1.0:
        dedt[:,0] = (self.E[:,1] - eupper)/(1*self.delta)

        duxdt = self.dvxdt(self.ux, self.uy,self.rho1, self.P1)
        duydt = self.dvydt(self.ux, self.uy,self.rho1, self.P1)

        # Find dt
        dt = self.dt(self.ux,self.uy,self.E,self.rho1,self.P1)

        # Boundary conditions
        # Update time for the variables
        self.rho1[:,:] = dt*drhodt + self.dx*(self.Gradient(self.rho1)[1] + self.Gradient(self.rho1)[0])/8.
        self.rho1[:,0] = -dPdy[:,0]/(self.F)        # BC for density, top
        self.rho1[:,-1] = -dPdy[:,-1]/(self.F)      # BC for density, bottom

        self.ux[:,:] = dt*duxdt + self.dx*(self.Gradient(self.ux)[1] + self.Gradient(self.ux)[0])/8.
        self.ux[0,:] = 0.0                          # BC for left
        self.ux[-1,:] = 0.0                         # BC for right

        self.uy[:,:] = dt*duydt + self.dx*(self.Gradient(self.uy)[1] + self.Gradient(self.uy)[0])/8.
        self.uy[:,0] = 0.0                          # BC for left
        self.uy[:,-1] = 0.0                         # BC for right

        self.E[:,:] = dt*dedt + self.dx*(self.Gradient(self.E)[1] + self.Gradient(self.E)[0])/8.
        # BC for energy
        if (abs(self.ux[:,-1])).any() < 1.0:
            self.E[:,-1] = dt*(self.E[:,-2] - elower)/self.delta

        if (abs(self.ux[:,0])).any() < 1.0:
            self.E[:,0] = dt*(self.E[:,1] - eupper)/self.delta

        self.P1[:,:] = self.E[:,:]/(self.gamma - 1)
        #print self.rho1[100,:]
        self.T1[:,:] = self.mu*self.m_u*self.P1/(self.k_B*self.rho1)
        self.T1[:,0] = 0.9*self.T1[:,0]             # BC for top
        self.T1[:,-1] = 1.1*self.T1[:,-1]           # BC for bottom

        return dt



    def SanityCheck1(self):
        """
        Sanity check for the gradient
        """
        x = np.linspace(0, 2*np.pi, self.nx)
        y = np.linspace(0, 2*np.pi, self.ny)
        yy,xx = np.meshgrid(x,y)

        print np.shape(xx), np.shape(yy)
        f = np.sin(xx) + np.sin(yy)
        grad = self.Gradient(f)

        fig1 = plt.figure('dfdy')
        ax1 = fig1.gca(projection='3d')
        ax1.plot_surface(xx, yy, grad[1]*self.dx, cmap=cm.coolwarm)
        plt.savefig('SC_grad_dfdy.png')

        fig2 = plt.figure('dfdx')
        ax2 = fig2.gca(projection='3d')
        ax2.plot_surface(xx, yy, grad[0]*self.dx, cmap=cm.coolwarm)
        plt.savefig('SC_grad_dfdx.png')



################################
N = 100

HD = HydroDynamics(N)

initial = HD.Set_Initial()
inte = HD.Integrate()
#sc1 = HD.SanityCheck1()

Vc = HD.ConvectiveVelocity()    # Convective velocity and differens from up and down
DelStar = HD.Del_star()         # Temp. gradient of the star
Fc = HD.ConvectiveFlux()        # Convective flux, mixing length and temp difference

# For visualisation
#step = HD.step()

#vis = FVis.FluidVisualiser()
#vis.save_data(100, HD.step, rho=HD.rho1, u=HD.ux,w=HD.uy, P=HD.P1, sim_fps=1.0)
#vis.animate_2D('rho1', matrixLike=False, extent=[0,12e+6,0,4e+6])#folder='FVis_output_2018-05-23_11-55'
#vis.delete_current_data()

plt.show()
