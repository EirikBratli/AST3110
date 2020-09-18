import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from EnergyProduction import EnergyProduction
from Project1 import CoreModel
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


class Star(EnergyProduction, CoreModel):

    def __init__(self,rho, T):
        """
        Set initial parameters, both form input and non-input.
        """
        self.N = int(4e+4)
        self.L0 = 1.0*LumSun                    # Initial Luminosity
        self.R0 = 1.0*SolarRadius *1.3          # New initial radius
        self.M0 = 1.0*SolarMass                 # Initial Mass
        self.rho0 = 1.42e-7*meanRhoSun          # Old initial density
        self.T0 = 5770                          # Initial temperature

        # Mass fractions
        self.X = 0.7
        self.Y3 = 1e-10
        self.Y = 0.29
        self.Y4 = self.Y - self.Y3
        self.Z = 0.01
        self.Z_Li7 = 1e-13
        self.Z_Be7 = 1e-13

        # Constants
        self.m_u = 1.660539e-27                          # kg
        self.m_p, self.m_e = 1.6726e-27, 9.1094e-31      # kg,kg
        self.e, self.k_B = 1.602e-19, 1.381e-23          # C, m^2kgs^-2K^-1
        self.eps0 = 8.954e-12                            # Fm^-1
        self.h = 6.627e-34; self.avo = 6.0221e+23         # Js, mol^-1
        self.c = 299792458                               # m/s
        self.sigma = 5.67e-8                                # Wm^-2 K^-4
        self.a = 4*self.sigma/float(self.c)
        self.G = 6.672e-11                                  # Nm^2 kg^-2
        self.delta = 1
        self.alpha = 1

        self.core = CoreModel(self.rho0,self.T0)
        Kvalues, Rvalues,Tvalues = core.readfile('opacity.txt')

        #self.kappa = 3.98 # for sanity check
        self.mu = core.mu(2,0.75,0.5)           # Picking up mu

        # Arrays:
        self.m = np.zeros(self.N)               # Mass
        self.r = np.zeros(self.N)               # Radius
        self.L = np.zeros(self.N)               # Luminosity
        self.T = np.zeros(self.N)               # Temperature
        self.P = np.zeros(self.N)               # Pressure
        self.rho = np.zeros(self.N)             # Density
        self.eps = np.zeros(self.N)             # Energy



    def Hp(self, m,r,T):
        """
        Compute the pressure height scale.
        """

        Hp = self.k_B*T/(self.g(m,r)*self.m_u*self.mu)
        return Hp

    def Cp(self,rho):
        """
        The heat capasity for constant pressure.
        """

        Cp = 5*self.k_B/(2*self.mu*self.m_u)
        return Cp

    def g(self, m, r):
        """
        Gravitational acceleration.
        """

        g = self.G*m/(r**2)
        return g

    def Grad_ad(self,P,T,rho):
        """
        Temperature gradient for adiabatic system.
        """

        Grad_ad = P*self.delta/(T*rho*self.Cp(rho))
        return Grad_ad

    def Grad_stable(self,m,r,P,L,T,rho,kappa):
        """
        Compute the temperature gradient for only radiation.
        """

        Grad_stable = 3.0*kappa*rho*self.Hp(m,r,T)*L/(64.0*np.pi*self.sigma*T**4*r**2)
        return Grad_stable

    def U(self,m,r,P,L,T,rho,kappa):
        """
        Compute the factor U.
        """

        U = 64*self.sigma*(T**3)/(3*kappa*self.Cp(rho)*(rho**2)) * np.sqrt(self.Hp(m,r,T)/(self.delta*self.g(m,r)))
        return (U)

    def xi_solver(self,m,r,P,L,T,rho,k):
        """
        Solve the cubic expression for xi.
        """

        self.lm = self.Hp(m,r,T)*self.alpha
        self.rp = self.lm
        self.S = 4*np.pi*self.rp**2
        self.Q = np.pi*(self.rp)**2
        self.d = 2*self.rp

        U = self.U(m,r,P,L,T,rho,k)

        self.A = U/self.lm**2
        self.B = U**2*self.S/(self.Q*self.d*self.lm**3)
        self.C = -self.A*(self.Grad_stable(m,r,P,L,T,rho,k) - self.Grad_ad(P,T,rho))

        # Solve for xi:
        coeffs = np.array([1.0,self.A,self.B, self.C])
        xi = np.roots(coeffs)
        for root in xi:
            if np.isreal(root):
               xi = root
               break

        xi = np.real(xi)
        return (xi)

    def Grad_star(self,m,r,P,L,T,rho,k):
        """
        Gradient for the star, outside of the parcel.
        """

        U = self.U(m,r,P,L,T,rho,k)
        lm = self.Hp(m,r,T)*self.alpha
        xi = self.xi_solver(m,r,P,L,T,rho,k)

        Grad_star = self.Grad_ad(P,T,rho) + xi**2 + xi*U*self.S/(self.Q*self.d*lm)
        return Grad_star


    def ConvectiveFlux(self,m,r,P,L,T,rho,kappa):
        """
        Compute the convective flux.
        """

        Hp = self.Hp(m,r,T)
        d = self.delta
        lm = Hp*self.alpha
        g = self.g(m,r)
        Cp = self.Cp(rho)
        xi = self.xi_solver(m,r,P,L,T,rho,kappa)

        FC = rho*Cp*T*np.sqrt(g*d)*Hp**(-3./2.)*(lm/2.)**2 * xi**3
        return FC

    def RadiationFlux(self,m,r,P,L,T,rho,kappa):
        """
        Compute the flux from radiation
        """

        FR = 16*self.sigma*T**4/(3*kappa*rho*self.Hp(m,r,T))*self.Grad_star(m,r,P,L,T,rho,kappa)
        return FR

    def TotFlux(self,r,L):
        """
        Compute the total flux, FR + FC
        """

        FrFc = L/(4*np.pi*r**2)
        return FrFc


    def dTdm_conv(self,m,r,T,rho,L,P,drdm,k):
        """
        calculate the differential of temperature with respect to mass, convert
        from dTdr to dTdm by multiply by drdm.
        """

        dTdr = -self.Grad_star(m,r,P,L,T,rho,k)*T/(self.Hp(m,r,T))
        dTdm = dTdr*drdm

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



    def Integrate_Conv(self, Tin, rhoin):

        """
        Integration: Calulate how the radius, pressure, density, lumiosity and
        temperature varies with mass. Integrate through the star including convection.
        Use variation in the step length of dm. Return mass, radius, luminosity,
        temperature, pressure, desity energy and the last index when m > 0.
        """


        print 'Integrate with variation in steplength and convection'

        core = CoreModel(self.rho0, self.T0)        # Call for the core model
        # Make parameters local variables
        m, r, P, L, T, rho = self.m, self.r, self.P, self.L, self.T, self.rho

        # Set initial conditions for the arrays.
        m[0] = self.M0; r[0] = self.R0
        L[0] = self.L0; self.T[0] = Tin
        P[0] = core.Pressure(rhoin,T[0])
        rho[0] = core.Density(P[0], T[0])
        k0 = core.Kappa(rho[0], T[0])

        E0 = EnergyProduction(rho[0], T[0])
        E0.Lambda()
        self.eps[0] = sum(E0.FusionRate())

        # Empty list for checking different goals and parameters.
        self.ii = []; rr = []; ddd = []; ll = []; ind_conv = []; tt = []

        # Defining arrays for the gradients and there initial conditions
        Del_star = np.zeros(self.N);Del_stable = np.zeros(self.N);Del_ad = np.zeros(self.N)
        Del_star[0] = self.Grad_star(m[0],r[0],P[0],L[0],T[0],rho[0],k0)
        Del_ad[0] = self.Grad_ad(P[0],T[0],rho[0])
        Del_stable[0] = self.Grad_stable(m[0],r[0],P[0],L[0],T[0],rho[0],k0)

        self.Fc = np.zeros(self.N); self.Fr = np.zeros(self.N); self.Ftot = np.zeros(self.N)
        self.Fc[0] = self.ConvectiveFlux(m[0],r[0],P[0],L[0],T[0],rho[0],k0)
        self.Fr[0] = self.RadiationFlux(m[0],r[0],P[0],L[0],T[0],rho[0],k0)
        self.kappalist = []
        convSize = []       # List for size of convetion zone

        self.PP1 = np.zeros(self.N)
        self.PP2 = np.zeros(self.N)

        # Euler
        for i in xrange(0, self.N-1):

            #Check if mass becomes negative
            if m[i] < 0.0:
                self.ii.append(i)
                break

            # update energy:
            E = EnergyProduction(rho[i], T[i])
            E.Lambda()
            self.eps[i] = sum(E.FusionRate())

            k = core.Kappa(rho[i],T[i])
            self.kappalist.append(k)

            # update differentials
            drdm = core.drdm(r[i], rho[i])
            dPdm = core.dPdm(r[i], m[i])
            dLdm = core.dLdm(self.eps[i], rho[i])

            # Update the temperature gradients
            Del_ad[i] = self.Grad_ad(P[i],T[i],rho[i])
            Del_stable[i] = self.Grad_stable(m[i],r[i],P[i],L[i],T[i],rho[i],k)

            # Check for convection
            if self.Grad_stable(m[i],r[i],P[i],L[i],T[i],rho[i],k) > self.Grad_ad(P[i],T[i],rho[i]):
                # Convetion happens
                Del_star[i] = self.Grad_star(m[i],r[i],P[i],L[i],T[i],rho[i],k)
                dTdm = self.dTdm_conv(m[i],r[i],T[i],rho[i],L[i],P[i],drdm,k)

                convSize.append(r[i])
                ind_conv.append(i)

                self.Fc[i] = self.ConvectiveFlux(m[i],r[i],P[i],L[i],T[i],rho[i],k)

            else:
                # Convection dont happnes
                Del_star[i] = Del_stable[i]
                dTdm = core.dTdm(rho[i], T[i], L[i], r[i])
                self.Fc[i] = 0

            self.Fr[i] = self.RadiationFlux(m[i],r[i],P[i],L[i],T[i],rho[i],k)
            self.Ftot[i] = self.Fr[i] + self.Fc[i]

            # Energy fractions:
            self.PP1[i] = E.PP1()
            self.PP2[i] = E.PP2()

            # update the step length
            dm = self.dm(r[i],drdm,P[i],dPdm,T[i],dTdm,L[i],dLdm)
            ddd.append(dm)          # Append dm's to list for evaluation if needed

            # Integration
            r[i+1] = r[i] + drdm*dm
            P[i+1] = P[i] + dPdm*dm
            L[i+1] = L[i] + dLdm*dm
            T[i+1] = T[i] + dTdm*dm
            rho[i+1] = core.Density(P[i+1], T[i+1])

            m[i+1] = m[i] + dm

            
            # print every nth interation
            if i%1000 == 0:
                print i,'-', m[i+1]/SolarMass, r[i+1]/r[0], L[i+1]/L[0]


            # Check radius of core
            if L[i+1] >= L[0]*0.995:
                ll.append(i)

            if r[i+1] <= 0.1*r[0]:
                rr.append(L[i+1]/L[0])
            if T[i+1] > 20e+6:
                tt.append(r[i+1])


        ### for loop end ###

        SizeConv = convSize[0] - convSize[-1]
        print 'Size of convection zone = %g of stellar radius'%(SizeConv/r[0])
        self.ind_conv = ind_conv[-1]    # Index for the where convection starts

        if r[max(ll)] > r[0]*0.1:
            print 'Size of core is bigger than 1/10 of R0 = %g'%(r[0]*0.1)
            print 'At 0.995x L0, R = %g at R0 = %g'%(r[max(ll)], r[0])
            print r[max(ll)]/r[0], 'relative size of the core w.r.t. radius'

        print tt[0]/r[0], 'Radius where T = 20MK'

        # Find the first index when m < 0:
        if len(self.ii) > 0:

            ind0 = min(self.ii)
        else:
            ind0 = self.N-1


        # Make variables global
        self.m = m; self.r = r; self.P = P; self.L = L; self.T = T; self.rho = rho
        self.ind0 = ind0; self.Del_star = Del_star; self.Del_ad = Del_ad
        self.Del_stable = Del_stable; self.ConvSize = convSize

        return self.m,self.r,self.P,self.L,self.T,self.rho,self.eps,self.ind0


    def SanityCheck(self,kappa, M0,R0,P0,T0,L0,rho0):
        """
        Check if the calculations are similar to a set of given values.
        """

        lm = self.Hp(M0,R0,T0)*self.alpha
        a = np.sqrt(self.g(M0,R0)*self.delta*lm**2/(4*self.Hp(M0,R0,T0)))
        v0 = a*self.xi_solver(M0,R0,P0,L0,T0,rho0)

        Fr = self.RadiativeFlux(M0,R0,P0,L0,T0,rho0)
        Fc = self.ConvectiveFlux(M0,R0,P0,L0,T0,rho0)
        totFlux = self.TotFlux(R0,L0)

        FrFtot = Fr/totFlux
        FcFtot = Fc/totFlux

        print 'Sanity check:'
        print '------------------------'
        print 'mu = %g'%(self.mu)
        print 'T = %g K'%(T0)
        print 'rho = %g kg/m^3'%(rho0)
        print 'R = %g R_sun'%(R0/SolarRadius)
        print 'M(R) = %g M_sun'%(M0/SolarMass)
        print 'kappa = %g m^2/kg'%(kappa)
        print 'alpha = %g'%(self.alpha)
        print '-----'

        print 'For radiavtive transport'

        print 'Grad_stable = %g'%(self.Grad_stable(M0,R0,P0,L0,T0,rho0))
        print 'Grad_ad = %g '%(self.Grad_ad(P0,T0,rho0))
        print '-----'

        print 'Hp = %g m'%(self.Hp(M0,R0,T0))
        print 'U = %e'%(self.U(M0,R0,P0,L0,T0,rho0))
        print 'xi = %e'%(self.xi_solver(M0,R0,P0,L0,T0,rho0))
        print 'Grad_star = %g'%(self.Grad_star(M0,R0,P0,L0,T0,rho0))
        print 'v = %g m/s'%(v0)
        print 'Grad_p = %g'%(self.Grad_star(M0,R0,P0,L0,T0,rho0) - self.xi_solver(M0,R0,P0,L0,T0,rho0)**2)

        print 'Relation between Fc and F_tot: %g'%(FcFtot)
        print 'Relation between Fr and F_tot: %g'%(FrFtot)

    def SanityPlot(self):
        """
        Plot the temperature gradients and the cross section of the star using
        the sanity check initial values.
        """
        print 'Plot for sanity check'
        print '------------------------'

        R_values = self.r[:self.ind0]/self.r[0]
        L_values = self.L[:self.ind0]/self.L[0]
        F_C_list = self.Fc[:self.ind0]
        n = len(R_values)
        R0 = R_values[0]
        show_every = 50
        core_limit = 0.995


        plt.figure('Gradients')

        plt.plot(R_values, self.Del_ad[:self.ind0],'-g',label=r'$\nabla_{ad}$')
        plt.semilogy(R_values, self.Del_stable[:self.ind0],'-r',label=r'$\nabla_{stable}$')
        plt.semilogy(R_values, self.Del_star[:self.ind0],'-b',label=r'$\nabla^{*}$')

        plt.title('Temperature Gradient')
        plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2,borderaxespad=0.)
        plt.xlabel(r'$R/R_{sun}$')
        plt.ylabel(r'log($\nabla$)')
        #plt.savefig('SanityCheck_gradients.png')
        plt.savefig('BestFit_Gradients.png')

        ##############
        # Gradient plot for the area with convection
        ConvSize = self.ConvSize

        ind_conv = int(self.ind_conv)
        plt.figure('Gradient zoomed')
        plt.plot(R_values[:ind_conv], self.Del_ad[:ind_conv],'-g',label=r'$\nabla_{ad}$')
        plt.semilogy(R_values[:ind_conv], self.Del_stable[:ind_conv],'-r',label=r'$\nabla_{stable}$')
        plt.semilogy(R_values[:ind_conv], self.Del_star[:ind_conv],'-b',label=r'$\nabla^{*}$')

        plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2)
        plt.xlabel(r'$R/R_{sun}$')
        plt.ylabel(r'log($\nabla$)')
        plt.savefig('Gradient_zoom.png')


        ##############

        plt.figure('X-section')
        fig = plt.gcf() # get current figure
        ax = plt.gca()  # get current axis
        rmax = 1.2*R0
        ax.set_xlim(-rmax,rmax)
        ax.set_ylim(-rmax,rmax)
        ax.set_aspect('equal')	# make the plot circular
        j = show_every
        for k in range(0, n-1):
        	j += 1
        	if j >= show_every:	# don't show every step - it slows things down
        		if(L_values[k] > core_limit):	# outside core
        			if(F_C_list[k] > 0.0):		# convection
        				circR = plt.Circle((0,0),R_values[k],color='red',fill=False)
        				ax.add_artist(circR)
        			else:				# radiation
        				circY = plt.Circle((0,0),R_values[k],color='yellow',fill=False)
        				ax.add_artist(circY)
        		else:				# inside core
        			if(F_C_list[k] > 0.0):		# convection
        				circB = plt.Circle((0,0),R_values[k],color='blue',fill = False)
        				ax.add_artist(circB)
        			else:				# radiation
        				circC = plt.Circle((0,0),R_values[k],color='cyan',fill = False)
        				ax.add_artist(circC)
        		j = 0
        circR = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='red',fill=True)		# These are for the legend (drawn outside the main plot)
        circY = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='yellow',fill=True)
        circC = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='cyan',fill=True)
        circB = plt.Circle((2*rmax,2*rmax),0.1*rmax,color='blue',fill=True)
        ax.legend([circR, circY, circC, circB], \
                    ['Convection outside core', 'Radiation outside core', 'Radiation inside core', 'Convection inside core'])
                    # only add one (the last) circle of each colour to legend
        plt.legend(loc=2)
        plt.title('Cross-section of star')

        #plt.savefig('SanityCheck_Xsection.png')
        plt.savefig('Xsection_100rho0_1T0_13r0.png')



        plt.show()


    def ParameterPlot(self):
        ind0 = self.ind0
        m,r,P,L,T,rho = self.m[:ind0],self.r[:ind0],self.P[:ind0],self.L[:ind0],self.T[:ind0],self.rho[:ind0]

        param = [r,m,P,L,T,rho]
        norm = [r[0], m[0], P[0], L[0], 1e+6, meanRhoSun]
        name = ['Radius','Mass', 'Pressure','Luminosity','Temperature','Density']
        label = [r'$R/R_{0}$',r'$M/M_{0}$', r'$P/P_{0}$',r'$L/L_{0}$',r'$T [MK]$',r'$\rho/\overline{\rho}_{\odot}$']

        for p in range(1,len(param)):
            plt.figure('%s' %(name[p]))
            if p != 2 and p != 5:
                print '%s'%(name[p])
                plt.plot(param[0]/r[0], param[p]/norm[p], '-b', label='%s'%(name[p]))

            else:
                print '%s'%(name[p])
                plt.semilogy(param[0]/r[0], param[p]/norm[p], '-b', label='%s'%(name[p]))

            #plt.legend(bbox_to_anchor=(0.01, 0.99), loc=2,borderaxespad=0.)
            plt.legend(loc=1)
            plt.title('Radius vs %s'%(name[p]))
            plt.xlabel(r'%s'%(label[0]))
            plt.ylabel(r'%s'%(label[p]))
            plt.savefig('Parameter_%svs%s.png'%(name[p],name[0]))

        plt.show()


    def FluxPlot(self):
        """
        Plot how the energy transport are through the star.
        """

        ind0 = self.ind0
        m,r,P,L,T,rho = self.m[:ind0],self.r[:ind0],self.P[:ind0],self.L[:ind0],self.T[:ind0],self.rho[:ind0]
        kappa = self.kappalist[:ind0]

        Fr = self.Fr[:ind0]; Fc = self.Fc[:ind0]; Ftot = self.Ftot[:ind0]

        plt.figure('flux')
        plt.plot(r/r[0], Fc/Ftot, '-b', label='Convetive flux')
        plt.plot(r/r[0], Fr/Ftot, '-r', label='Radiation flux')

        plt.legend(bbox_to_anchor=(0.,1.01,1., 0.1), loc=3, ncol=2, mode='expand')
        plt.xlabel(r'$R/R_{sun}$')
        plt.ylabel(r'Frational energy transport, $F_{i}/F_{tot}$')
        plt.savefig('FluxPlot.png')

        plt.show()

    def EnergyPlot(self):
        """
        Relative energy production from the PP chains, one from PP-1 and one from PP-2.
        Overploted by the total relative energy produced in the core.
        """

        r, eps = self.r[:self.ind0], self.eps[:self.ind0]
        eps_max = max(eps)
        pp1 = self.PP1[:self.ind0]; pp2 = self.PP2[:self.ind0]

        plt.figure('Energy')
        plt.plot(r/r[0], pp1/eps, '-b', label=r'PP-1/$\epsilon$')
        plt.plot(r/r[0], pp2/eps, '-r', label=r'PP-2/$\epsilon$')
        plt.plot(r/r[0], eps/eps_max, '-g', label=r'$\epsilon/\epsilon_{max}$')

        plt.legend(bbox_to_anchor=(0.,0.9, 1, 0.1), loc=3, ncol=3, mode='expand')
        plt.xlabel(r'$R/R_0$')
        plt.ylabel('Energy')
        plt.ylim(-0.05,1.15)
        plt.savefig('EnergyProduction.png')

        plt.show()



############### Call the functions: ################
# Initial values for sanity check
T0 = 0.9e+6
rho0 = 55.9
R0 = 0.84*SolarRadius
M0 = 0.99*SolarMass
L0 = LumSun
kappa = 3.98
alpha = 1

# Initial values
core = CoreModel(rho0,T0)
mu = core.mu(2,0.75,0.5)
P0 = core.Pressure(rho0,T0)

T = 5770                                # Initial temperature
rho = 1.42e-7*meanRhoSun *100           # New initial density


star = Star(rho,T)
integrateConv = star.Integrate_Conv(T,rho)
#sanity = star.SanityCheck(kappa, M0,R0,P0,T0,L0,rho0)
#SanityPlot = star.SanityPlot()
#ParmPlot = star.ParameterPlot()
#flux = star.FluxPlot()
energy = star.EnergyPlot()



"""
Terminal >>> SanityCheck:
Sanity check:
------------------------
mu = 0.616333
T = 900000 K
rho = 55.9 kg/m^3
R = 0.84 R_sun
M(R) = 0.99 M_sun
kappa = 3.98 m^2/kg
alpha = 1
-----
For radiavtive transport
Grad_stable = 3.1724
Grad_ad = 0.4
-----
Hp = 3.15953e+07 m
U = 602602
xi = 1.187263e-03
Grad_star = 0.400001
v = 65.4188 m/s
Grad_p = 0.4
Relation between Fc and F_tot: 0.873912
Relation between Fr and F_tot: 0.126088
"""
