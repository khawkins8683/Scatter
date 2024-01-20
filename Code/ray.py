import numpy as np

## -------------------------------------------------------------------------------------------------------------------------------------
##Classes
## -------------------------------------------------------------------------------------------------------------------------------------


class Ray:
    def __init__(self, k, r, wavelength):
        self.k = k          #intermediate ks can be calculated from the position vectors
        self.r = np.array([r])
        self.wavelength = wavelength
        self.OPL = np.array([0])
        self.Q = np.array([ np.identity(3) ])
        self.PRT = np.array([ np.identity(3) ])
    
    def OPLCumulative(self):
        return np.sum(self.OPL)
    
