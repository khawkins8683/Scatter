import functools 
import numpy as np
import utility as u

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
    def __init__(self, k, r, wavelength, stokeIn = np.array([1,0,0,0])):
        self.k = k          #intermediate ks can be calculated from the position vectors
        self.r = np.array([r])#         units in microns
        self.wavelength = wavelength# units of microns
        self.OPL = np.array([0])
        self.transmission = np.array([1])
        self.horizontal = np.array([u.CreateSP(np.array([0,0,1]),k)[0]])
        self.Q = np.array([ np.identity(3) ])
        self.jones = np.array([ np.identity(2) ])
        self.PRT = np.array([ np.identity(3) ])
        self.MM = np.array([ np.identity(4) ])
        self.MMLocal = np.array([np.identity(4)] )
        self.stokes = np.array([ stokeIn ])

    def __str__(self):
        return 'Ray obj: \nlength: %f'%(len(self.r))       

#methods
# Cumulative calculations OPL/PRT/Q
    def OPLCumulative(self):
        return np.sum(self.OPL)
    
    def transmissionCumulative(self):
        return np.prod(self.transmission)
    
    def QCumulative(self):
        return functools.reduce( np.dot, self.Q )
    
    def PRTCumulative(self):
        return functools.reduce( np.dot, self.PRT )
    
    def MMCumulative(self):
        return functools.reduce( np.dot, self.MM )        
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
#update jones/calculate jones
    def updateJonesSPFresnel(self,kIn, eta, mat1,mat2, mode):
        n1 = mat1.n + 1j *mat1.k
        n2 = mat2.n + 1j *mat2.k
        thetai = u.vectorAngle(kIn , eta)
        thetat = u.SnellsLaw(n1,n2,thetai)
        #thetat = u.vectorAngle(self.k , eta)
        #thetai = u.SnellsLaw(mat2.n,mat1.n,thetat)
        #print("kvec in ",self.k," in mat ",mat1)
        #print("Theta t: ",thetat/u.deg)
        #print("Theta i: ",thetai/u.deg)
        if mode == "REFLECT":
            rs = (n1*np.cos(thetai) - n2*np.cos(thetat)  ) / (n1*np.cos(thetai) + n2*np.cos(thetat)  )
            rp = (n2*np.cos(thetai) - n1*np.cos(thetat)  ) / (n2*np.cos(thetai) + n1*np.cos(thetat)  )
            jones = np.array([[rs,0],[0,rp]])
            
        elif mode == "REFRACT":
            
            ts = 2*n1*np.cos(thetai) / (n1*np.cos(thetai) + n2*np.cos(thetat)  )
            tp = 2*n1*np.cos(thetai) / (n2*np.cos(thetai) + n1*np.cos(thetat)  )
            jones = np.array([[ts,0],[0,tp]])
        else:
            print("Mode error --- updateJonesSP ")

        #print("Jones: ",jones)
        self.jones = np.append(self.jones , np.array([jones]), axis = 0)
        return jones   
     
    #some scattered jones may not be created by scatter!
    def updateJonesSPScatter(self):
        #here => if material is is use identity if 
        jones = np.identity(2)
        self.jones = np.append(self.jones , np.array([jones]), axis = 0)
        return jones

#geometry updates
    def updateHorizontal(self,eta):#*note k should be updated before this is called!!!!!!!!
        h = u.CreateSP(eta,self.k)[0]#np.dot(self.Q[-1], self.horizontal[-1] ) #np.cross(kin,self.k)
        self.horizontal = np.append(self.horizontal , np.array([h]), axis = 0)
        return h   

    def updateQMatrix(self,eta,kin):#*note k should be updated before this is called!!!!!!!!
        q = u.PRT(np.identity(2), eta,kin, self.k)
        self.Q = np.append(self.Q , np.array([q]), axis = 0)
        return q    

#polarization matrix updates - based ON JONES matrix
    def updatePRTMatrix(self,eta,kin):#*note k should be updated before this is called!!!!!!!!
        prt = u.PRT(self.jones[-1], eta,kin, self.k)
        self.PRT = np.append(self.PRT, np.array([prt]), axis = 0)
        return prt  
    
    #TODO => separate  the rotation from the MM creation
    def updateMMFromJones(self):#*note k should be updated before this is called!!!!!!!!
        #first get theta
        #we need a transmission factor here
        mm = np.real(u.JonesToMueller(self.jones[-1]))
        self.MMLocal = np.append(self.MMLocal , np.array([mm]), axis = 0)
        return mm


    def updateMMFromJonesFresnel(self,mode,eta,mat1,mat2):#*note k should be updated before this is called!!!!!!!!
        #first get theta
        #we need a transmission factor here
        if mode == "REFLECT":
            factor = 1
        elif mode == "REFRACT":
            thetat = u.vectorAngle(self.k , eta)#assuming this is kout
            thetai = u.SnellsLaw(mat2.n,mat1.n,thetat)
            factor = (mat2.n*np.cos(thetat)) / (mat1.n*np.cos(thetai))
        mm = factor * np.real(u.JonesToMueller(self.jones[-1]))
        self.MMLocal = np.append(self.MMLocal , np.array([mm]), axis = 0)
        return mm

    def updateGlobalMM(self):
        mm = self.MMLocal[-1]
        theta = u.vectorAngle(self.horizontal[-2],self.horizontal[-1])
        mmRotated = u.RotateMueller(mm,theta)
        self.MM = np.append(self.MM , np.array([mmRotated]), axis = 0)
        return mmRotated
