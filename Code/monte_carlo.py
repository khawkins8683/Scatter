import math
import matplotlib.pyplot as plt
import numpy as np
import pySCATMECH as scatmech
from pySCATMECH.fresnel import *
from . import ray as r
from . import utility as u

#stack = scatmech.fresnel.FilmStack()
#glass = scatmech.fresnel.OpticalFunction(1.5)
#air = scatmech.fresnel.OpticalFunction(1)
#def rs(mat1,mat2,wavelength,aoi):
#    [[rs,rps],[rps,rp]]=stack.reflectionCoefficient(20*deg,0.5,air,glass)


#now we need a function to randomly sample phase functions
#integrate the CDF

def RayleighJones(thetaScat): # I don't think we need phi scat just yet
    jm = np.array([[np.cos(thetaScat),0],[0,1]])
    return jm

## now we need to do a 'ray' trace
## set up material thickness
## set up material properties
## TODO - add debug printing
## TODO - add in prt - jones from fresnel
## TODO - adding ray paths together/ making BSDF from ray paths

def MonteCarloTrace(RayPath, mat1, DiffuseMaterial, debug= False):
    thickness = DiffuseMaterial.thickness
    # Step 1 -> refract ray into the material
    #update rayQ and jones and stuff
    kin = RayPath.k#store input k for updates 
    #step 1 update k,r,and eta
    eta = np.array([0,0,1])
    RayPath.k = RayPath.refract(eta, mat1, DiffuseMaterial)
    RayPath.r = np.append(RayPath.r , np.array([np.array([0,0,0])]), axis = 0)
    RayPath.updateJonesSPFresnel(eta, mat1,DiffuseMaterial, "REFRACT")
    #now update the ray polarization properties
    RayPath.updateHorizontal(kin)
    RayPath.updateOPL(mat1)
    RayPath.updateTransmission(mat1)
    RayPath.updateQMatrix(eta,kin)
    RayPath.updatePRTMatrix(eta,kin)
    RayPath.updateMMMatrix()
    #now that we are in the material we will propagate the ray until a z component is either 0<z<thickness
    live = True;i = 1
    while live:
        i += 1
        kin = RayPath.k
        #step  1.1 get prop distance 
        r = DiffuseMaterial.sampleDistance()*RayPath.k + RayPath.r[-1]
        #check to see if we leave the material
        #update ray r and ray k and eta -> jons
        if r[-1]<0 or r[-1]>thickness:
            if debug: print("CHECK TRiggered")
            if r[-1]<0:#see if retro reflection occurs
                # prop ray to the edge of the material
                sign=-1
                live = RayPath.interface(DiffuseMaterial,mat1,0,sign)
                if debug: print("Retro-REFLECT Out of material? ",live)
            if r[-1]>thickness:#see if ray transmits
                sign=1
                live = RayPath.interface(DiffuseMaterial,mat1,thickness,sign)
                if debug: print("Transmits OUT of material? ",live)
            mode = "REFLECT" if live else "REFRACT"#internal TIR vs 
            eta = np.array([0,0,sign])
            RayPath.updateJonesSPFresnel(eta, DiffuseMaterial, mat1, mode)
        else:
            pass
            # get scatter k direction
            [theta,phi] = DiffuseMaterial.phaseFunction(RayPath)#theta and phi are relative to the ray
            live = RayPath.scatter(theta,phi,r)#just use previously calculated r
            #here we also should update the jones matrix => isotropic => identity
            RayPath.updateJonesSPScatter()
            eta = u.ScatterEta(kin,RayPath.k)#get eta halfway between kin and RayPath.scatter
        #TODO - update more ray parameters
        #TODO - update opl
        RayPath.updateHorizontal(kin)
        RayPath.updateOPL(DiffuseMaterial)
        RayPath.updateTransmission(DiffuseMaterial)
        RayPath.updateQMatrix(eta,kin)
        RayPath.updatePRTMatrix(eta,kin)
        RayPath.updateMMMatrix()
        #update Mueller Matrix
    return RayPath

def MonteCarloTrial(N,mat1,mat2,wavelength,kIn, debug= False):
    if debug: i=0
    rayList = []
    for i in range(N):
        if debug: print( "Tracing Ray i: ", i )
        rayList.append( MonteCarloTrace( r.Ray(kIn, np.array([0,0,0]), wavelength ), mat1, mat2, debug  ) )
    return rayList
