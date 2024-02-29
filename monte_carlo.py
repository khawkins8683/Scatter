import math
import matplotlib.pyplot as plt
import numpy as np
import pySCATMECH as scatmech
from pySCATMECH.fresnel import *
import ray as r
import utility as u

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

def MonteCarloTrace(inputs):
    [RayPath, mat1, DiffuseMaterial, debug] = inputs
    thickness = DiffuseMaterial.thickness
    # Step 1 -> refract ray into the material
    #update rayQ and jones and stuff
    kin = RayPath.k#store input k for updates 
    #step 1 update k,r,and eta
    eta = np.array([0,0,1])
    RayPath.k = RayPath.refract(eta, mat1, DiffuseMaterial)
    RayPath.r = np.append(RayPath.r , np.array([np.array([0,0,0])]), axis = 0)
    RayPath.updateJonesSPFresnel(kin,eta, mat1,DiffuseMaterial, "REFRACT")
    RayPath.updateMMFromJonesFresnel("REFRACT",eta,mat1,DiffuseMaterial)
    #now update the ray polarization properties
    RayPath.updateHorizontal(eta)
    RayPath.updateOPL(mat1)
    RayPath.updateTransmission(mat1)
    RayPath.updateQMatrix(eta,kin)
    RayPath.updatePRTMatrix(eta,kin)
    RayPath.updateGlobalMM()
    #now that we are in the material we will propagate the ray until a z component is either 0<z<thickness
    live = True;i = 1
    while live:
        i += 1
        kin = RayPath.k
        if debug: print("Next MonteCarlo step: ",i)
        #step  1.1 get prop distance 
        r = DiffuseMaterial.sampleDistance()*RayPath.k + RayPath.r[-1]
        #check to see if we leave the material
        #update ray r and ray k and eta -> jones -> mueller
        if r[-1]<=0 or r[-1]>=thickness:
            # prop ray to the edge of the material
            if debug: print("CHECK TRiggered")
            if r[-1]<=0:#see if retro reflection occurs
                sign=-1;thickness=0
            else:#see if ray transmits
                sign=1;thickness = DiffuseMaterial.thickness               

            live = RayPath.interface(DiffuseMaterial,mat1,thickness,sign)#updates k and r
            if debug: print("Leave Material? ",live,RayPath.r[-1],RayPath.k,[sign,thickness])
            mode = "REFLECT" if live else "REFRACT"#internal TIR vs 
            eta = np.array([0,0,sign])
            RayPath.updateJonesSPFresnel(kin,eta, DiffuseMaterial, mat1, mode)#mode for reflectino/refraction coeffs
            RayPath.updateMMFromJonesFresnel(mode,eta,DiffuseMaterial,mat1)
            if debug: print("Surface updated jm/mm: ",i)
        else:
            # get scatter k direction
            [theta,phi] = DiffuseMaterial.phaseFunction(RayPath)#theta and phi are relative to the ray
            live = RayPath.scatter(theta,phi,r)#just use previously calculated r#updates k and r
            eta = u.ScatterEta(kin,RayPath.k)#get eta halfway between kin and RayPath.scatter
            
            #here we also should update the jones matrix => isotropic => identity
            #this needs to have a material input
            ## get the jones / MM from the material
            if DiffuseMaterial.calculus == "jones":
                #update the stored jones matrix => then mueller
                if debug: print("scatter updated jm/mm: ",i)
                RayPath.updateJonesSPScatter()
                RayPath.updateMMFromJones()
            elif DiffuseMaterial.calculus == "mueller":
                #zero jones and update mueller
                pass
            else:
                print("calculus type error: ",DiffuseMaterial.calculus=="jones")
            
            
        #here we have => k/r/eta/jones/mm-stokes
        #TODO - update more ray parameters
        #TODO - update opl
        #non - pol + local to global updates
        
        RayPath.updateOPL(DiffuseMaterial)
        RayPath.updateTransmission(DiffuseMaterial)
        RayPath.updateQMatrix(eta,kin)
        RayPath.updateHorizontal(eta)
        #for depolarizing MMS => this could be difficult
        #certain materials need to not convert the Jones => i.e ray input
        RayPath.updatePRTMatrix(eta,kin)
        RayPath.updateGlobalMM()
        #update Mueller Matrix
        if i > 1000:
            print("Infinite: trace -- abort: ")
            live = False
    return RayPath

def MonteCarloTrial(N,mat1,mat2,wavelength,kIn, debug= False):
    if debug: i=0
    rayList = []
    for i in range(N):
        if debug: print( "Tracing Ray i: ", i )
        rayList.append( MonteCarloTrace( [r.Ray(kIn, np.array([0,0,0]), wavelength ), mat1, mat2, debug  ]) )
    return rayList
