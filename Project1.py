import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from EnergyProduction import EnergyProduction
import sys
import math

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


# Solar Parameters:
Lum0 = 1.0*LumSun
R0 = 0.72*SolarRadius
M0 = 0.8*SolarMass
rho0 = 5.1*meanRhoSun
T0 = 5.7e+6

dm = -SolarMass*1e-4


class CoreModel(EnergyProduction):
    def __init__(self, rho, Temp):
        """
        Initial values and parameters. Define the arrays for the integration and
        some usefull constants for the calcualtions.
        """
        # Nucleus parameters, mass fractions:
        self.X, self.Y, self.Z = 0.7, 0.29, 0.01
        self.Y3, self.Z_Li7, self.Z_Be7 = 1e-10, 1e-13, 1e-13
        self.Y4 = self.Y-self.Y3

        # Inputparameters:
        self.rho0 = rho
        self.T0 = Temp
        self.N = int(1e+4)

        # Constants:
        self.m_u = 1.660539e-27                          # kg
        self.m_p, self.m_e = 1.6726e-27, 9.1094e-31      # kg,kg
        self.e, self.k_B = 1.602e-19, 1.382e-23          # C, m^2kgs^-2K^-1
        self.eps0 = 8.954e-12                            # Fm^-1
        self.h = 6.627e-34; self.avo = 6.0221e+23         # Js, mol^-1
        self.c = 299792458                               # m/s
        self.sigma = 5.67e-8                                # Wm^-2 K^-4
        self.a = 4*self.sigma/float(self.c)
        self.G = 6.672e-11                                  # Nm^2 kg^-2

        # Arrays:
        self.m = np.zeros(self.N)               # Mass
        self.r = np.zeros(self.N)               # Radius
        self.L = np.zeros(self.N)               # Luminosity
        self.T = np.zeros(self.N)               # Temperature
        self.P = np.zeros(self.N)               # Pressure
        self.rho = np.zeros(self.N)             # Density
        self.eps = np.zeros(self.N)             # Energy




    def readfile(self, filename):
        """
        Read the opacity.txt file and put the logarithmic values of R, T and kappa
        into arrays, and returns the arrays.
        """

        f = open(filename, 'r')
        firstline = f.readline()
        args = firstline.split()
        logRstrs = args[1:]
        self.Rvalues = []

        # Put the logR values in a list and transform to array
        for ind, arg in enumerate(args):
            if ind > 0:
                arg = float(arg)
                self.Rvalues.append(arg)
        self.Rvalues = np.array(self.Rvalues[:])

        line2 = f.readline()
        values = np.zeros((70,20))

        # Put the values of first column in a logT array, and the rest in a
        # kappa array.
        for i,l in enumerate(f):
            lines = l.split()
            for j,arg in enumerate(lines):
                values[i,j] = float(arg)

        self.Kvalues = values[:,1:]                # cm^2/g
        self.Tvalues = values[:,0]
        return self.Kvalues, self.Rvalues, self.Tvalues

        # Extrapolation:
    def Kappa(self, rho, Temp):
        """
        Convert rho and T into logarithmic values. If these values are outside
        the range of the listed values, then need to extrapolate. Use interpolation
        to find kappa values. Returs the corresponding kappa value for the input
        rho and temperature. The extraploation does not work properly, but not
        needed for possitive mass.
        """

        # Inputparameters
        self.rhoIn = rho
        self.TempIn = Temp
        Kvalues, Rvalues,Tvalues = self.readfile('opacity.txt')

        rho = self.rhoIn                     # g/cm^3        to cgs units
        Temp = self.TempIn

        # Turn the inputparameters to logarithmic values
        R = rho/float((Temp/(1e+6))**3)/1e+3         # Input R
        logTemp = np.log10(Temp)                # Input T
        logR = np.log10(R)                      # log of input R
        """
        # Check for in range of logR and logT:
        if min(self.Rvalues) >= logR:
            print 'R = %.2f is out of range listed R values, need to extrapolate'%logR
            #logR = min(self.Rvalues)
        if logR >= max(self.Rvalues):
            print 'R = %.2f is out of range listed R values, need to extrapolate'%logR
            #logR = max(self.Rvalues)
        if min(self.Tvalues) >= logTemp:
            print 'T = %.2f is out of range listed T values, need to extrapolate'%logTemp
            #logTemp = min(self.Tvalues)
        if logTemp >= max(self.Tvalues):
            print 'T = %.2f is out of range listed T values, need to extrapolate'%logTemp
            #logTemp = max(self.Tvalues)
        """
        
        # Interpolate 2D
        f = interpolate.interp2d(self.Rvalues, self.Tvalues, self.Kvalues)

        # Compute intepolated kappa values
        kappaNew = f(logR, logTemp)

        #K2 = f2(logR, logTemp)                     # for ekstraploated values
        self.kappa = 10**(kappaNew)/10.
        #self.kappa = 10**(K2)/10                   # for ekstraploated values
        return self.kappa


    def mu(self, nx, ny, nz):
        """
        Calculate the mean molecular mass, mu.
        """
        mu = 1.0/(float(nx*self.X + ny*self.Y + nz*self.Z))
        return mu

    def Pressure(self, rho, T):
        """
        Calulate the pressure as a function of the density and temperature. Use
        both graviational pressure and radiation pressure.
        """
        self.P0 = rho*self.k_B*T/(self.mu(2,0.75,0.5)*self.m_u)    # Gravitational pressure
        self.P0 += self.a*T**4/3.                    # add radiation pressure
        return self.P0

    def Density(self, P, T):
        """
        Calulate the density as a function of pressure and temperature. Need to
        subtract the pressure from radiation.
        """
        #self.P = P

        self.rho0 = self.mu(2,0.75,0.5)*self.m_u*P/(self.k_B*T)
        self.rho0 -= self.mu(2,0.75,0.5)*self.m_u*self.a*T**3/(3.0*self.k_B)
        return self.rho0


    def drdm(self,r,rho):
        """
        Differetial for r
        """

        drdm = 1.0/(4.0*np.pi*r**2*rho)
        return drdm

    def dPdm(self,r, m):
        """
        Differential for P
        """

        dPdm = -self.G*m/(4.0*np.pi*r**4)
        return dPdm

    def dLdm(self, eps, rho):
        """
        Differential for L
        """

        dLdm = eps/rho
        return dLdm

    def dTdm(self,rho,T,L,r):
        """
        Differential for T
        """

        dTdm = -3.0*self.Kappa(rho,T)*L/(256.0*(np.pi)**2*self.sigma*r**4*T**3)

        return dTdm

    def dm(self, r,drdm,P,dPdm,T,dTdm,L,dLdm):
        """
        Calculation the dynamic step sizes, through dm = pV/f, where V is r, L,
        T and P, and f are drdm, dPdm, dLdm and dTdm. Return the lowest values
        when the sign are taken care of.
        """

        p = 1e-3
        dm1 = p*r/drdm
        dm2 = p*P/dPdm
        dm3 = p*L/dLdm
        dm4 = p*T/dTdm
        dmm = [abs(dm1),abs(dm2),abs(dm3), abs(dm4)]

        dm = -min(dmm)
        return dm

    def Integrate(self, M0, R0, P0, T0, rho0, L0,dm):
        """
        Integration: Calulate how the radius, pressure, density, lumiosity and
        temperature varies with mass. And returns the arrays of mass, radius,
        pressure, density, lumiosity and temperature, together with the last
        index for m > 0
        """


        print 'Integrate for:'

        self.M0 = M0; self.R0 = R0; self.L0 = L0

        # differential arrays
        drdm = np.zeros(self.N)
        dPdm = np.zeros(self.N)
        dLdm = np.zeros(self.N)
        dTdm = np.zeros(self.N)



        # Set initial conditions for the arrays.
        self.m[0] = self.M0; self.r[0] = self.R0;
        self.L[0] = self.L0; self.T[0] = self.T0;
        self.rho[0] = self.Density(P0, self.T0)
        self.P[0] = self.Pressure(self.rho0, self.T0)

        drdm[0] = 1.0/(4.0*np.pi*self.R0**2*self.rho[0])
        dPdm[0] = -self.G*self.M0/(4.0*np.pi*self.R0**4)
        dTdm[0] = -3.0*self.Kappa(self.rho[0],self.T[0])*self.L0\
                    /(256.0*(np.pi)**2*self.sigma*self.R0**4*self.T0**3)

        E0 = EnergyProduction(self.rho[0], self.T0)
        E0.Lambda()
        self.eps[0] = sum(E0.FusionRate())
        dLdm[0] = (self.eps[0])

        print '-----'
        ii = []
        # Euler
        for i in np.arange(0, self.N-1):

            # check if m becomes negative
            if self.m[i] < 0:
                ii.append(i)
                break

            # update energy:
            E = EnergyProduction(self.rho[i], self.T[i])
            E.Lambda()
            self.eps[i] = sum(E.FusionRate())

            # update differentials
            drdm[i] = self.drdm(self.r[i], self.rho[i])
            dPdm[i] = self.dPdm(self.r[i], self.m[i])
            dLdm[i] = self.dLdm(self.eps[i], self.rho[i])
            dTdm[i] = self.dTdm(self.rho[i], self.T[i], self.L[i], self.r[i])


            # Interation
            self.m[i+1] = self.m[i] + dm

            self.r[i+1] = self.r[i] + drdm[i]*dm
            self.P[i+1] = self.P[i] + dPdm[i]*dm
            self.L[i+1] = self.L[i] + dLdm[i]*dm
            self.T[i+1] = self.T[i] + dTdm[i]*dm
            self.rho[i+1] = self.Density(self.P[i+1], self.T[i+1])


            #if i%500 == 0:      # print every 500 interation
                #print drdm[i+1]*dm, self.r[i+1]/self.SolarRadius
                #print dTdm[i+1]*dm, self.T[i+1]*1e-6
                #print dLdm[i+1]*dm, self.L[i+1]/self.L0
                #print dPdm[i+1]*dm, self.P[i+1]/PressureConv0, self.rho[i+1]
                #print self.rho[i+1], self.P[i+1], self.r[i+1]
                #print dmm, dm




        if len(ii) > 0:
            ind0 = min(ii)
        else:
            ind0 = self.N-1

        """
        plt.figure(1)
        plt.plot(self.m[:ind0]/SolarMass, self.r[:ind0]/SolarRadius)
        plt.xlabel(r'$M/M_{sun}$', fontsize=9)
        plt.ylabel(r'$R/R_{sun}$', fontsize=9)
        plt.title('Radius vs Mass')
        plt.savefig('RvsM.png')

        plt.figure(2)
        plt.plot(self.m[:ind0]/SolarMass, self.L[:ind0]/Lum0)
        plt.xlabel(r'$M/M_{sun}$', fontsize=9)
        plt.ylabel(r'$L/L_{sun}$', fontsize=9)
        plt.title('Luminosity vs Mass')
        plt.savefig('LvsM.png')

        plt.figure(3)
        plt.plot(self.m[:ind0]/SolarMass, self.T[:ind0]*1e-6)
        plt.xlabel(r'$M/M_{sun}$', fontsize=9)
        plt.ylabel(r'T [MK]', fontsize=9)
        plt.title('Temperature vs Mass')
        plt.savefig('TvsM.png')

        plt.figure(4)
        plt.semilogy(self.m[:ind0]/SolarMass, self.rho[:ind0]/meanRhoSun)
        plt.xlabel(r'$M/M_{sun}$', fontsize=9)
        plt.ylabel(r'$\rho/\rho_{sun}$', fontsize=9)
        plt.title('Density vs Mass')
        plt.savefig('RhovsM.png')

        plt.show()
        """

        return self.m, self.r, self.P,self.L,self.T,self.rho, ind0




    def VarIntegrate(self, M0, Rin, Pin, Tin, rhoin, L0):
        """
        Integration: Calulate how the radius, pressure, density, lumiosity and
        temperature varies with mass. Integrate from outer rim of the convection
        zone to the core. Use variation in the step length of dm. Return mass,
        radius, luminosity, temperature, pressure, desity energy and the last index
        when m > 0.
        """


        print 'Integrate with variation in steplength, for'

        self.M0 = M0; self.L0 = L0

        # differential arrays
        drdm = np.zeros(self.N)
        dPdm = np.zeros(self.N)
        dLdm = np.zeros(self.N)
        dTdm = np.zeros(self.N)

        # Set initial conditions for the arrays.
        self.m[0] = self.M0; self.r[0] = Rin
        self.L[0] = self.L0; self.T[0] = Tin
        self.P[0] = self.Pressure(rhoin, Tin)
        self.rho[0] = self.Density(Pin, Tin)    # If not ok, try the other, should do the same
        #self.rho[0] = self.Density(self.P[0], self.T[0])

        E0 = EnergyProduction(rhoin, self.T[0])
        E0.Lambda()
        self.eps[0] = sum(E0.FusionRate())

        drdm[0] = 1.0/(4.0*np.pi*Rin**2*rhoin)
        dPdm[0] = -self.G*self.M0/(4.0*np.pi*Rin**4)
        dLdm[0] = (self.eps[0])
        dTdm[0] = -3.0*self.Kappa(self.rho[0],self.T[0])*self.L0\
                    /(256.0*(np.pi)**2*self.sigma*Rin**4*Tin**3)

        # Print initial values

        # Empty list for checking different goals and parameters.
        self.ii = []; rr = []; ddd = []; ll = []

        # Euler
        for i in xrange(0, self.N-1):

            # update energy:
            E = EnergyProduction(self.rho[i], self.T[i])
            E.Lambda()
            self.eps[i] = sum(E.FusionRate())

            # update differentials
            drdm[i] = self.drdm(self.r[i], self.rho[i])
            dPdm[i] = self.dPdm(self.r[i], self.m[i])
            dLdm[i] = self.dLdm(self.eps[i], self.rho[i])
            dTdm[i] = self.dTdm(self.rho[i], self.T[i], self.L[i], self.r[i])

            # update the step length
            dm = self.dm(self.r[i],drdm[i],self.P[i],dPdm[i],self.T[i],dTdm[i],self.L[i],dLdm[i])
            ddd.append(dm)          # Append dm's to list for evaluation if needed

            #Check if mass becomes negative
            if self.m[i] < 0:
                self.ii.append(i)
                break

            # Integration
            self.m[i+1] = self.m[i] + dm

            self.r[i+1] = self.r[i] + drdm[i]*dm
            self.P[i+1] = self.P[i] + dPdm[i]*dm
            self.L[i+1] = self.L[i] + dLdm[i]*dm
            self.T[i+1] = self.T[i] + dTdm[i]*dm
            self.rho[i+1] = self.Density(self.P[i+1], self.T[i+1])

            #if i%100 == 0:      # print every nth interation
                #print drdm[i+1]*dm, self.r[i+1]/self.SolarRadius
                #print dTdm[i+1]*dm, self.T[i+1]*1e-6
                #print dLdm[i+1]*dm, self.L[i+1]/self.L0
                #print dPdm[i+1]*dm, self.P[i+1]/PressureConv0, self.rho[i+1]
                #print self.m[i+1], self.r[i+1], drdm[i]#, self.rho[i+1]#, self.T[i+1], self.P[i+1]

            # Check radius of core
            if self.L[i+1] >= self.L[0]*0.995:
                ll.append(i)

            if self.r[i+1] <= 0.1*self.r[0]:
                rr.append(self.L[i+1]/self.L[0])

        if self.r[max(ll)] > self.r[0]*0.1:
            print 'Size of core is bigger than 1/10 of R0 = %g'%(self.r[0]*0.1)
            print 'At 0.995x L0, R = %g at R0 = %g'%(self.r[max(ll)], self.r[0])


        if len(self.ii) > 0:
            # Find the first index when m < 0:
            ind0 = min(self.ii)
        else:
            ind0 = self.N-1

        """
        # Plot for the final model
        plt.figure(1)
        plt.plot(self.r[:ind0]/SolarRadius, self.m[:ind0]/SolarMass)
        plt.xlabel(r'$R/R_{sun}$'); plt.ylabel(r'$M/M_{sun}$')
        #plt.savefig('MofR.png')

        plt.figure(2)
        plt.plot(self.r[:ind0]/SolarRadius, self.L[:ind0]/Lum0)
        plt.xlabel(r'$R/R_{sun}$'); plt.ylabel(r'$L/L_{sun}$')
        #plt.savefig('LofR.png')

        plt.figure(3)
        plt.plot(self.r[:ind0]/SolarRadius, self.T[:ind0]*1e-6)
        plt.xlabel(r'$R/R_{sun}$'); plt.ylabel(r'$T [MK]$')
        #plt.savefig('TofR.png')

        plt.figure(4)
        plt.semilogy(self.r[:ind0]/SolarRadius, self.eps[:ind0])
        plt.xlabel(r'$R/R_{sun}$'); plt.ylabel(r'$\epsilon$')
        #plt.savefig('epsofR.png')

        plt.figure(5)
        plt.semilogy(self.r[:ind0]/SolarRadius, self.rho[:ind0]/meanRhoSun)
        plt.xlabel(r'$R/R_{sun}$'); plt.ylabel(r'$\rho/\rho_{sun}$')
        #plt.savefig('rhoofR.png')

        plt.figure(6)
        plt.semilogy(self.r[:ind0]/SolarRadius, self.P[:ind0]/self.P[0])
        plt.xlabel(r'$R/R_{sun}$'); plt.ylabel(r'$P/P_{0}$')
        #plt.savefig('PofR.png')
        """

        #plt.show()


        return self.m, self.r, self.P,self.L,self.T,self.rho, self.eps, ind0



    def DiffInitValue(self, R0, T0, P0, rho0):
        """
        Calls the integration function, choose ether with og wothout  varying
        setplength. Plots how R, P, T and rho variates with different initial
        values.
        """

        # Set of different parameters
        Rval = [0.2*R0,1./3.*R0,0.5*R0, 0.75*R0,R0,1.5*R0,2*R0,4*R0]
        Tval = [0.2*T0,1./3.*T0,0.5*T0,0.75*T0,T0,1.5*T0,2*T0,4*T0]
        Pval = [0.2*P0,1./3.*P0,0.5*P0,0.75*P0,P0,1.5*P0,2*P0,4*P0]
        rhoval = [0.2*rho0,1./3.*rho0,0.5*rho0,0.75*rho0,rho0,1.5*rho0,2*rho0,4*rho0]

        # values tested
        values = ['0.2', '1/3', '0.5','0.75', '1.0', '1.5', '2.0', '4.0']
        var = ['R_0', 'T_0', 'P_0', r'\rho_0']

        for i in range(len(Rval)):
            print values[i], rhoval[i]
            # varying R:
            m, r, P, L, T, rho,eps, ind0 = self.VarIntegrate(M0,Rval[i],PressureConv0,\
                T0,rhoConv0,Lum0)

            # Varying T:
            #m, r, P, L, T, rho,eps,ind0 = self.VarIntegrate(M0,R0,PressureConv0,\
                #Tval[i],rhoConv0,Lum0)

            # Varying P
            #m, r, P, L, T, rho,eps,ind0 = self.VarIntegrate(M0,R0,Pval[i],T0,\
            #    rhoConv0,Lum0)

            # Varying rho
            #m, r, P, L, T, rho,eps, ind0 = self.VarIntegrate(M0,R0,PressureConv0,
            #                            T0,rhoval[i],Lum0)



            plt.figure(1)
            plt.plot(m[:ind0]/SolarMass, r[:ind0]/SolarRadius, label=r'$%s %s$'%(values[i],var[2]))
            plt.xlabel(r'$M/M_{sun}$', fontsize=8)
            plt.ylabel(r'$R/R_{sun}$', fontsize=8)
            plt.legend(bbox_to_anchor=(0.8, 0.8), loc=2,borderaxespad=0.)
            #plt.savefig('Varyrho_forR.png')
            #plt.savefig('VaryT_forR.png')
            #plt.savefig('VaryP_forR.png')
            #plt.savefig('VaryR_forR.png')

            plt.figure(2)
            plt.plot(m[:ind0]/SolarMass, T[:ind0]*1e-6, label=r'$%s %s$'%(values[i],var[2]))
            plt.xlabel(r'$M/M_{sun}$', fontsize=8)
            plt.ylabel(r'$T [MK]$', fontsize=8)
            plt.legend(bbox_to_anchor=(0.8, 0.8), loc=2, borderaxespad=0.)
            #plt.savefig('Varyrho_forT.png')
            #plt.savefig('VaryT_forT.png')
            #plt.savefig('VaryP_forT.png')
            #plt.savefig('VaryR_forT.png')

            plt.figure(3)
            plt.plot(m[:ind0]/SolarMass, L[:ind0]/Lum0, label=r'$%s %s$'%(values[i],var[2]))
            plt.xlabel(r'$M/M_{sun}$', fontsize=8)
            plt.ylabel(r'$L/L_{sun}$', fontsize=8)
            plt.legend(bbox_to_anchor=(0.8, 0.8), loc=2, borderaxespad=0.)
            #plt.savefig('Varyrho_forL.png')
            #plt.savefig('VaryT_forL.png')
            #plt.savefig('VaryP_forL.png')
            #plt.savefig('VaryR_forL.png')

            plt.figure(4)
            plt.semilogy(m[:ind0]/SolarMass, rho[:ind0]/meanRhoSun, label=r'$%s %s$'%(values[i],var[2]))
            plt.xlabel(r'$M/M_{sun}$', fontsize=8)
            plt.ylabel(r'$\rho/\rho_{conv0}$', fontsize=8)
            plt.legend(bbox_to_anchor=(0.8, 0.8), loc=2, borderaxespad=0.)
            #plt.savefig('Varyrho_for_rho.png')
            #plt.savefig('VaryT_for_rho.png')
            #plt.savefig('VaryP_for_rho.png')
            #plt.savefig('VaryR_for_rho.png')

        plt.tight_layout()
        #plt.show()

        return m, r, P, L, T, rho



    def RofM(self, rho):
        """
        Ploting r(m) when there is no dynamic step size for differnt values of dm.
        """

        factors = np.array([0.25,0.5,1.0,1.5,2.0,3.0])
        dmlist = -SolarMass*factors*1e-4

        for ind, dm in enumerate(dmlist):


            # Calls the integration function
            m, r, P, L, T, rho, ind0 = core.Integrate(M0,R0,PressureConv0, T0,\
                                        rhoConv0, Lum0, dm)

            plt.figure(1)     # Plots
            plt.plot(r/SolarRadius, m/SolarMass, label=r'$\partial m=%e$'%dm)


        plt.title(r'Different dm, $dm_0=%g$'%(dmlist[2]))
        plt.xlabel(r'$R/R_{sum}$')
        plt.ylabel(r'$M/M_{sun}$')
        plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2, borderaxespad=0.)
        plt.savefig('Different_dm.png')
        #plt.show()

    def ToZero(self):
        """
        Change R0, T0, rho0 to find where m, r and L goes to zero. Need to run
        the triple for-loop first to find good fitting values to the parameters.
        Plot ether in the for-loop or the outside are for finding the best values,
        the outside are fitted for the best values.
        """

        Tvalues = np.linspace(2.*T0, 2.1*T0, 4)
        Rvalues = np.linspace(0.5*R0, 0.65*R0, 4)
        rhovalues = np.linspace(0.85*rho0, 1.1*rho0, 4)
        Pvalues = PressureConv0#*np.arange([0.5,1.0, 0.05])

        # Choose to variate R,T and rho.
        """
        RR,TR,rhoR,RL,TL,rhoL = [],[],[],[],[],[]

        # Find the best fitting initial values for m, r and L to go to zero.

        for i in range(len(Tvalues)):
            for j in range(len(Rvalues)):
                for k in range(len(rhovalues)):
                    print i,j,k

                    #m, r, P, L, T, rho,eps, ind0 = core.VarIntegrate(M0,R0,Pvalues[i],\
                                                #Tvalues[i], rhovalues[i], Lum0)

                    #m, r, P, L, T, rho,eps, ind0 = core.VarIntegrate(M0,r,Pvalues[i],\
                                                #t, rhoConv0, Lum0)

                    #m, r, P, L, T, rho, eps,ind0 = core.VarIntegrate(M0,Rvalues[i],\
                                                #Pvalues[i],T0, rhovalues[i], Lum0)

                    m, r, P, L, T, rho, eps,ind0 = core.VarIntegrate(M0,Rvalues[j],\
                                                PressureConv0,Tvalues[i], rhovalues[k], Lum0)


                    if m[-1]/SolarMass <= 0.05:
                        if r[-1]/SolarRadius <= 0.05:
                            RR.append(Rvalues[j])
                            TR.append(Tvalues[i])
                            rhoR.append(rhovalues[k])
                            Rbest = [min(RR),min(TR),min(rhoR)]
                        if L[-1]/Lum0 <= 0.05:
                            RL.append(Rvalues[j])
                            TL.append(Tvalues[i])
                            rhoL.append(rhovalues[k])
                            Lbest = [min(RL),min(TL),min(rhoL)]

                        if r[-1]/SolarRadius <= 0.05:
                            plt.figure(1)
                            plt.plot(m[:ind0]/SolarMass, r[:ind0]/SolarRadius, label='%dT,%dR,%drho'%(i,j,k))
                            plt.legend(bbox_to_anchor=(0.55, 1.05), ncol=2, loc=2, borderaxespad=0.)
                            plt.xlabel(r'$M/M_{sun}$')
                            plt.ylabel(r'$R/R_{sun}$')

                        if L[-1]/Lum0  <= 0.05:
                            plt.figure(2)
                            plt.plot(m[:ind0]/SolarMass, L[:ind0]/Lum0, label='%dT,%dR,%drho'%(i,j,k))
                            plt.legend(bbox_to_anchor=(0.55, 1.05), ncol=2, loc=2, borderaxespad=0.)
                            plt.xlabel(r'$M/M_{sun}$')
                            plt.ylabel(r'$L/L_{sun}$')


            print i,j,k


        print Rbest
        print Lbest
        """

        # Call using the best fit parameters. Best so far
        m, r, P, L, T, rho, eps, ind0 = core.VarIntegrate(M0,R0*0.6,\
                                    PressureConv0,T0*2.1, rho0*0.85, Lum0)

        Mbest = m[-1]/SolarMass; Rbest = r[-1]/SolarRadius; Lbest = L[-1]/Lum0
        #print Mbest,Rbest,Lbest




        plt.figure(1)
        plt.plot(m[:ind0]/SolarMass, r[:ind0]/SolarRadius,\
        label=r'$%.3fR_{0}, %.2fT_{0}, %.2f\rho_{0}$'%(Rvalues[0]/R0, Tvalues[1]/T0, rhovalues[3]/rho0))

        plt.legend(bbox_to_anchor=(0.57, 0.08), loc=2, borderaxespad=0.)
        plt.xlabel(r'$M/M_{sun}$')
        plt.ylabel(r'$R/R_{sun}$')
        plt.title(r'Best fit R, $R_{min}=%.3f, M_{min}=%.3f$'%(Rbest,Mbest))
        #plt.savefig('bestfitR.png')

        plt.figure(2)

        plt.plot(m[:ind0]/SolarMass, L[:ind0]/Lum0,\
        label=r'$%.3fR_{0}, %.2fT_{0}, %.2f\rho_{0}$'%(Rvalues[0]/R0, Tvalues[1]/T0, rhovalues[3]/rho0))

        plt.legend(bbox_to_anchor=(0.57, 0.08), loc=2, borderaxespad=0.)
        plt.xlabel(r'$M/M_{sun}$')
        plt.ylabel(r'$L/L_{sun}$')
        plt.title(r'Best fit L, $L_{min}=%.3f, M_{min}=%.3f$'%(Lbest,Mbest))
        #plt.savefig('bstefitL.png')
        #"""

        #plt.show()

        self.Rbest = Rvalues[1]/R0; self.Tbest = Tvalues[0]/T0
        self.rhobest = rhovalues[3]/rho0
        return self.Rbest, self.Tbest, self.rhobest




# Calling the different functions:

core = CoreModel(rhoConv0, TempConv0)
read = core.readfile('opacity.txt')
kappa = core.Kappa(rhoConv0, TempConv0)
mu = core.mu(2,0.75,0.5)

#p = core.Pressure(rhoConv0, T0)
#rho = core.Density(PressureConv0, T0)
#mass, radius, pressure, lumin, temp, rho, ind0 = core.Integrate(M0,R0, PressureConv0,\
                            #T0, rhoConv0,Lum0, dm)         # ok
#mass, radius, pressure, lumin, temp, rho, eps,ind0 = core.VarIntegrate(M0,R0,PressureConv0,\
                        #T0, rhoConv0, Lum0)

# the final plots for model of the core
#mass, radius, pressure, lumin, temp, rho,eps,ind0 = core.VarIntegrate(M0,R0*0.5667,PressureConv0,\
#                        T0*2, rhoConv0*0.9, Lum0)

#VarInit = core.DiffInitValue(R0, T0, PressureConv0, rhoConv0)
#RofM = core.RofM(rho0)
#toZero = core.ToZero()
