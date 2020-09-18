"""
Solar parameters for the project. Used by called in the commando line with
Project1.py
"""
# Solar Parameters:
meanRhoSun = 1.408e+3           # kg m^-3
LumSun = 3.846e+26              # W = J s^-1
SolarMass = 1.989e+30           # kg
SolarRadius = 6.96e+8           # m
SolarLife = 4.57e+9             # yr
SolarKappa = 1                  # m^2 kg^-1

# Core
rhoCore = 1.62e+5                # kg m^-3
TempCore = 1.57e+7               # K
PressureCore =  3.45e+16         # Pa = N m^-2

# Bottom convection zone
rhoConv0 = 7.2e+3               # kg m^-3
TempConv0 = 5.7e+6              # K
PressureConv0 = 5.2e+14         # N m^-2
RadiusConv0 = 0.72*SolarRadius

# Solar Photoshere
rhoPhotos = 2e-4                # kg m^-3
TempPhotos = 5778               # K
PressurePhotos = 1.8e+8         # N m^-2
X_Photos = 0.7346
Y_Photos = 0.2485
Z_Photos = 0.017

# Solar Corona
rhoCorona = 1e-11               # kg m^-3
TempCorona = 1.5e+6             # K
PressureCorona = 0.1            # N m^-2
