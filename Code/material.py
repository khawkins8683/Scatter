import numpy as np

#exp typ material distance utility
def ProbDistExp( x, beta ):
    prob = (1/beta)*np.exp(-x / beta)
    return prob

def ExpSample(xRand,beta):
    xSample = -beta*np.log(1-xRand)
    return xSample

def IsotropicPhaseFunction(N=1):
    theta = np.random.rand(N) * np.pi
    phi = np.random.rand(N) * 2*np.pi
    return [theta, phi]

#scatterers
#first is Elsastic/Isotropic
#parameters = {"mean": 1.4E-6} 
#also needs a phase function

class Material:
    def __init__(self, n, thickness, scat):
        self.n = np.real(n)
        self.k = np.imag(n)
        self.scatParams = scat
        self.thickness = thickness

    def sampleDistance(self):
        #note this assumes self type is exp
        d = ExpSample( np.random.rand() , self.scatParams.mean )
        return d
    
    #run this to check for absobption 
    def scatterProb(self):
        return True
    
    #run this to get the new scatter direction 
    #todo:: do we have all the information?
    def phaseFunction(self,ray):
        return IsotropicPhaseFunction(1)

    
# ------------- notes --------------------------
    # white matter


#TODO:: make some specific materials

#axons 1-3 um diameter
#cover by axolemma
#myelinated sheath - insulator - its WHITE!
#higher refractive index 
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7063473/
#This 50% decrease in scattering power along the myelinated axons
#Furthermore, this directional dependence in scattering power and overall light attenuation did not occur in the gray matter regions where the myelin organization is nearly random.

#spinal coord => highly organized

#also unmyelinnated

#also unmyelinnated