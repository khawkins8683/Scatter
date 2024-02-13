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
        self.horizontal = np.array([np.array([1,0,0])])
        self.Q = np.array([ np.identity(3) ])
        self.jones = np.array([ np.identity(2) ])
        self.PRT = np.array([ np.identity(3) ])
        self.MM = np.array([ np.identity(4) ])
    

# Cumulative calculations OPL/PRT/Q
    def OPLCumulative(self):
        return np.sum(self.OPL)
    
    def transmissionCumulative(self):
        return np.prod(self.transmission)
    
    def QCumulative(self):
        return functools.reduce( np.dot, self.Q )
#utility    
    def refract(self,eta,mat1,mat2):
        return u.Refract3D(mat1.n,mat2.n, eta, self.k)
    
    def TIRQ(self,eta,mat1,mat2):
        return u.TIRCheck(mat1.n,mat2.n,eta,self.k)

#update internal parameters r/k
    def scatter(self,theta,phi,r):#update k and r
        #updates k and r values
        #returns true for continued scattering
        k = u.KScatter(self.k,theta,phi)
        self.k=k
        self.r= np.append(self.r , np.array([r]), axis = 0)
        return True
    
    def interface(self,mat1,mat2,zValue,sign):#update k and r
        #mat1 is current material
        #update ray k and ray r values
        #return boolean for continued prop in material
        r = propRayToEdge(self,zValue,sign)
        k = self.refract(np.array([0,0,sign]),mat1, mat2)
        live = self.TIRQ(np.array([0,0,sign]),mat1,mat2)
        self.k=k
        self.r= np.append(self.r , np.array([r]), axis = 0)
        return live
#update other
    def updateOPL(self,mat):
        distance = np.linalg.norm(self.r[-1]-self.r[-2])#should be in meters
        opl = mat.n * distance
        self.OPL = np.append(self.OPL , np.array([opl]), axis = 0)
        return opl

    def updateTransmission(self,mat):
        distance = np.linalg.norm(self.r[-1]-self.r[-2])#should be in meters
        abs = np.exp(  -2*np.pi/(self.wavelength/1E6) * mat.k * distance )#conver wl to meters
        self.transmission = np.append(self.transmission , np.array([abs]), axis = 0)
        return abs

    def updateJonesSPFresnel(self, eta, mat1,mat2, mode):
        n1 = mat1.n + 1j *mat1.k
        n2 = mat2.n + 1j *mat2.k
        thetat = u.vectorAngle(self.k , eta)
        if mode is "REFLECT":
            thetai = thetat
            rs = (n1*np.cos(thetai) - n2*np.cos(thetai)  ) / (n1*np.cos(thetai) + n2*np.cos(thetai)  )
            rp = (n2*np.cos(thetai) - n1*np.cos(thetai)  ) / (n1*np.cos(thetai) + n2*np.cos(thetai)  )
            jones = np.array([[rs,0],[0,rp]])
        elif mode is "REFRACT":
            thetai = u.SnellsLaw(mat2.n,mat1.n,thetat)
            ts = 2*n1*np.cos(thetai) / (n1*np.cos(thetai) + n2*np.cos(thetat)  )
            tp = 2*n1*np.cos(thetai) / (n2*np.cos(thetai) + n1*np.cos(thetat)  )
            jones = np.array([[ts,0],[0,tp]])
        else:
            print("Mode error --- updateJonesSP ")

        self.jones = np.append(self.jones , np.array([jones]), axis = 0)
        return jones    

    def updateJonesSPScatter(self):
        jones = np.identity(2)
        self.jones = np.append(self.jones , np.array([jones]), axis = 0)
        return jones

    def updateHorizontal(self,kin):#*note k should be updated before this is called!!!!!!!!
        h = np.cross(kin,self.k)
        self.horizontal = np.append(self.horizontal , np.array([h]), axis = 0)
        return h   



    def updateQMatrix(self,eta,kin):#*note k should be updated before this is called!!!!!!!!
        q = u.PRT(np.identity(2), eta,kin, self.k)
        self.Q = np.append(self.Q , np.array([q]), axis = 0)
        return q    

    def updatePRTMatrix(self,eta,kin):#*note k should be updated before this is called!!!!!!!!
        prt = u.PRT(self.jones[-1], eta,kin, self.k)
        self.PRT = np.append(self.Q , np.array([prt]), axis = 0)
        return prt  
    
    def updateMMMatrix(self):#*note k should be updated before this is called!!!!!!!!
        #first get theta
        theta = u.vectorAngle(self.horizontal[-2],self.horizontal[-1])
        mm = u.JonesToMueller(self.jones[-1])
        mmRotated = u.RotateMueller(mm,theta)
        self.MM = np.append(self.MM , np.array([mmRotated]), axis = 0)
        return mmRotated 