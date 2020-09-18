import numpy as np
import matplotlib.pyplot as plt

class EnergyProduction:
    def __init__(self, rho, Temp):
        """
        Different constants, make the input parameters into global variables, and
        calculate the densities for the particles in the reactions.
        """
        # Solar constants, mass fractions
        X, Y, Z = 0.7, 0.29, 0.01
        Y3, Z_Li7, Z_Be7 = 1e-10, 1e-13, 1e-13
        Y4 = Y-Y3

        # Constants
        self.m_u = 1.660539e-27                          # kg
        self.m_p, self.m_e = 1.6726e-27, 9.1094e-31      # kg,kg
        self.e, self.k_B = 1.602e-19, 1.382e-23          # C, m^2kgs^-2K^-1
        self.eps0 = 8.954e-12                            # Fm^-1
        self.h = 6.627e-34                               # Js
        self.avo = 6.0221e+23                            # mol^-1
        self.c = 299792458                               # m/s

        # Input Parameters
        self.rho = rho
        self.T = Temp
        # Parameters
        self.EnergyConv = 1.60217662e-13                # eV to Joule
        self.MassConv = 0.00100794*1e+6/(self.avo)      # kg/mol /mol^-1

        # Densities
        self.n_p = self.rho*X/(self.m_u)
        self.D = self.rho*X/(self.m_u)
        self.n_He3 = self.rho*Y3/(3.*self.m_u)
        self.n_He4 = self.rho*Y4/(4.*self.m_u)
        self.n_Li7 = self.rho*Z_Li7/(7.*self.m_u)
        self.n_Be7 = self.rho*Z_Be7/(7.*self.m_u)
        self.n_e = self.rho*(X + 2./3.*Y3 + 2./4.*Y4)/(self.m_u)


    def Lambda(self):
        """
        Calculate the lambda values for the different reactions, and returns the
        lambdas.
        """

        T9 = self.T/float(1e+9)             # Normalized T to billion Kelvin
        T9star1 = T9/(1+4.95e-2*T9)
        T9star2 = T9/(1+0.759*T9)

        # Lambdas
        self.lmbd11 = 4.01e-15*T9**(-2./3.)*np.exp(-3.380*T9**(-1./3.)) \
                    *(1+0.123*T9**(1./3.)+1.09*T9**(2./3.)+0.938*T9)/(self.avo*1e+6)

        self.lmbd33 = 6.04e+10*T9**(-2./3.)*np.exp(-12.276*T9**(-1./3.)) \
                    *(1+0.034*T9**(1./3.)-0.522*T9**(2./3.)-0.124*T9+ \
                    0.353*T9**(4./3.)+0.213*T9**(-5./3.))/(self.avo*1e+6)

        self.lmbd34 = 5.61e+6*T9star1**(5./6.)*T9**(-3./2.)*np.exp(-12.826\
                    *T9star1**(-1./3))/(self.avo*1e+6)

        self.lmbde7 = 1.34e-10*T9**(-1./2.)*(1-0.537*T9**(1./3.)+3.86*T9**(2./3.)\
                    +0.0027*T9**(-1)*np.exp(2.515e-3*T9**(-1)))/(self.avo*1e+6)

        self.lmbd17dash = (1.096e+9*T9**(-2./3.)*np.exp(-8.472*T9**(-1./3.))\
                    -4.830e+8*T9star2**(5./6.)*T9**(-3./2.)*np.exp(-8.472*T9star2**(-1./3.))\
                    +1.06e+10*T9**(-3./2.)*np.exp(-30.442*T9**(-1)))/(self.avo*1e+6)

        self.lmbd17 = (3.11e+5*T9**(-2./3.)*np.exp(-10.262*T9**(-1./3.)) \
                    + 2.53e+3*T9**(-3./2.)*np.exp(-7.306/float(T9)))/(self.avo*1e+6)

        return self.lmbd11, self.lmbd33,self.lmbd34,self.lmbde7,self.lmbd17,self.lmbd17dash


    def FusionRate(self):
        """
        Use the densities and the lambdas to compute the reation rate for all the
        different reactions. Then find and return the energy production.
        """

        # proton + proton
        self.r11 = self.lmbd11*self.n_p**2/(2.*self.rho)
        # He3 + He3
        self.r33 = self.lmbd33*self.n_He3*self.n_He3/(2.*self.rho)
        if self.r33 >= self.r11:
            self.r33 = self.r11
        # He3 + He4
        self.r34 = self.lmbd34*self.n_He3*self.n_He4/(self.rho)
        if self.r34 >= self.r33+self.r11:
            self.r34 = self.r33+self.r11
        # Be7 + e-
        self.re7 = self.lmbde7*self.n_e*self.n_Be7/(self.rho)
        if self.re7 >= self.r34:
            self.re7 = self.r34
        # Li7 + proton
        self.r17dash = self.lmbd17dash*self.n_p*self.n_Li7/(self.rho)
        if self.r17dash >= self.re7:
            self.r17dash = self.re7
        #Be7 + proton
        self.r17 = self.lmbd17*self.n_p*self.n_Be7/(self.rho)
        if self.r17 >= self.r34:
            self.r17 = self.r33

        # Energy from the reactions
        self.Q11 = 0.15+1.02    # MeV
        self.Q21 = 5.49         # MeV
        self.Q33 = 12.86        # MeV
        self.Q34 = 1.59         # MeV
        self.Qe7 = 0.05         # Mev
        self.Q17dash = 17.35    # MeV
        self.Q17 = 0.14         # MeV

        # Total Energy Out:



        self.E11 = self.r11*(self.Q11+self.Q21)*self.rho*self.EnergyConv
        self.E33 = self.r33*self.Q33*self.rho*self.EnergyConv
        self.E34 = self.r34*self.Q34*self.rho*self.EnergyConv
        self.Ee7 = self.re7*self.Qe7*self.rho*self.EnergyConv
        self.E17dash = self.r17dash*self.Q17dash*self.rho*self.EnergyConv
        self.E17 = self.r17*self.Q17*self.rho*self.EnergyConv

        return self.E11, self.E33, self.E34,self.Ee7,self.E17dash, self.E17

    def PP1(self):
        """
        Contribution of energy from PP-I chain.
        """

        PP1 = self.E11 + self.E33
        return PP1 *(self.r33)/(self.r34+self.r33)

    def PP2(self):
        """
        Contribution of energy from the PP-II chain.
        """

        PP2 = self.E11 + (self.E34 + self.Ee7 + self.E17dash)
        return PP2 *(self.r34)/(self.r33+self.r34)




#rhoCore = 1.62e+5                # kg m^-3
#TempCore = 10e+7               # K
#E = EnergyProduction(rhoCore, TempCore)
#l = E.Lambda()
#F = E.FusionRate()
#print F
