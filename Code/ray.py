import functools 
import numpy as np
from . import utility as u

def propRayToEdge(RayPath,zValue,sign):
    deltaZ = (sign*-1)*(RayPath.r[-1][-1] - zValue) #-thickness for transmission
    cosTheta = np.dot(np.array([0,0,sign]), RayPath.k)
    dPrime = deltaZ/cosTheta
    r = dPrime*RayPath.k + RayPath.r[-1]
    r[2] = zValue#avoid rounding errors
    return r

## -------------------------------------------------------------------------------------------------------------------------------------
##Classes
## -------------------------------------------------------------------------------------------------------------------------------------

#TODO -- add transmission parameter
class Ray:
    def __init__(self, k, r, wavelength):
        self.k = k          #intermediate ks can be calculated from the position vectors
        self.r = np.array([r])#         units in microns
        self.wavelength = wavelength# units of microns
        self.OPL = np.array([0])
        self.transmission = np.array([1])
        self.Q = np.array([ np.identity(3) ])
        self.PRT = np.array([ np.identity(3) ])
    

    # Cumulative calculations OPL/PRT/Q
    def OPLCumulative(self):
        return np.sum(self.OPL)
    
    def transmissionCumulative(self):
        return np.prod(self.transmission)
    
    def QCumulative(self):
        return functools.reduce( np.dot, self.Q )
    
    def refract(self,eta,mat1,mat2):
        return u.Refract3D(mat1.n,mat2.n, eta, self.k)
    
    def TIRQ(self,eta,mat1,mat2):
        return u.TIRCheck(mat1.n,mat2.n,eta,self.k)
    
    def updateTransmission(self,mat):
        distance = np.linalg.norm(self.r[-1]-self.r[-2])#should be in meters
        abs = np.exp(  -2*np.pi/(self.wavelength/1E6) * mat.k * distance )#conver wl to meters
        self.transmission = np.append(self.transmission , np.array([abs]), axis = 0)


    def scatter(self,theta,phi,r):
        #updates k and r values
        #returns true for continued scattering
        k = u.KScatter(self.k,theta,phi)
        self.k=k
        self.r= np.append(self.r , np.array([r]), axis = 0)
        return True
    
    def interface(self,mat1,mat2,zValue,sign):#mat1 is current material
        #update ray k and ray r values
        #return boolean for continued prop in material
        r = propRayToEdge(self,zValue,sign)
        k = self.refract(np.array([0,0,sign]),mat1, mat2)
        live = self.TIRQ(np.array([0,0,sign]),mat1,mat2)
        self.k=k
        self.r= np.append(self.r , np.array([r]), axis = 0)
        return live