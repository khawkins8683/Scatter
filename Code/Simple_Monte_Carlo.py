import functools 
import math

import matplotlib.pyplot as plt
import numpy as np
import pySCATMECH as scatmech
from pySCATMECH.fresnel import *

thresh = 1E-15
deg = np.pi/180
stack = scatmech.fresnel.FilmStack()
glass = scatmech.fresnel.OpticalFunction(1.5)
air = scatmech.fresnel.OpticalFunction(1)



def rs(mat1,mat2,wavelength,aoi):
    [[rs,rps],[rps,rp]]=stack.reflectionCoefficient(20*deg,0.5,air,glass)


def ProbDist( x, beta ):
    prob = (1/beta)*np.exp(-x / beta)
    return prob


def ExpSample(xRand,beta):
    xSample = -beta*np.log(1-xRand)
    return xSample

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

def MonteCarloTrace(RayPath, DiffuseMaterial, debug= False):
    thickness = DiffuseMaterial.thickness
    # Step 1 -> refract ray into the material
    kMat = Refract3D(1.0,  DiffuseMaterial.n , np.array([0,0,1]), RayPath.k , debug)
    RayPath.k = kMat
    #now update the ray polarization properties

    #now that we are in the material we will propagate the ray until a z component is either 0<z<thickness
    live = True
    i = 1
    while live:
        i += 1
        #step  1.1 get prop distance -- TODO: put the exp distribution as a type in the material
        d = ExpSample( np.random.rand() , DiffuseMaterial.gamma )#TODO think about doing a random seed for reproducibility
        
        #now prop ray d
        r = d*RayPath.k + RayPath.r[-1]

        #check 
        z = r[-1]
 
        if z<0 or z>thickness:
            if debug: print("CHECK TRiggered!!!!!")
             #TODO --- do not kill rays if they TIR
            if z<0:
                if debug: print("REFLECT")
                deltaZ = RayPath.r[-1][-1] - 0 #-thickness for transmission
                cosTheta = np.dot(np.array([0,0,-1]), RayPath.k)
                dPrime = deltaZ/cosTheta
                r = dPrime*RayPath.k + RayPath.r[-1]
                r[np.abs(r) < thresh ] = 0
                k = Refract3D(DiffuseMaterial.n, 1 , np.array([0,0,-1]), RayPath.k )
                live = TIRCheck(DiffuseMaterial.n,1.0,np.array([0,0,-1]),RayPath.k)
            if z>thickness:
                if debug: print("Transmit")
                deltaZ = thickness - RayPath.r[-1][-1] #-thickness for transmission
                cosTheta = np.dot(np.array([0,0,1]), RayPath.k)# eta switches sign for other side of material
                dPrime = cosTheta*deltaZ
                r = dPrime*RayPath.k + RayPath.r[-1]
                k = Refract3D(DiffuseMaterial.n, 1 , np.array([0,0,1]), RayPath.k )
                live = TIRCheck(DiffuseMaterial.n,1.0,np.array([0,0,1]),RayPath.k)
        else:
            pass
            # get scatter k direction
            [theta,phi] = IsotropicPhaseFunction(1)
            k = KScatter( RayPath.k ,theta,phi)
        #check again
        z = r[-1];    
        if z<0 or z>thickness:
            print("ERRROR:: ---------")
            print("live: ",live)
            print("kOut",k)
            print("kIn",RayPath.k)
            print('rOut',r)
            print('rIN', RayPath.r[-1])
            print("D: ",d)
            print("D`: ",dPrime)
            print("dZ: ",deltaZ)


        #now update ray parameters
        RayPath.k = k
        RayPath.r = np.append(RayPath.r , np.array([r]), axis = 0)
        #if i > 1000:    live = False
        
    return RayPath

def MonteCarloTrial(N,Mat,wavelength,kIn, debug= False):
    if debug: i=0
    rayList = []
    for i in range(N):
        if debug: i+=1
        if debug: print( "Tracing Ray i: ", i )
        rayList.append( MonteCarloTrace( Ray(kIn, np.array([0,0,0]), wavelength ), Mat, debug  ) )
    return rayList





#----------------------------------------------------------------------------------------------------------------------------------------
## Plotting functions

def PlotRayPath(Ray, slice = 1 ,ax = plt):
    positions = Ray.r
    #x values are z
    xVals = positions[:,-1]
    yVals = positions[:,slice]

    ax.plot(xVals, yVals,  marker='o')# label='line with marker'

def PlotMaterial(Mat,scale = 5, ax=plt):
    t = Mat.thickness
    ax.vlines(0,-scale*t,scale*t, colors='black')
    ax.vlines(t,-scale*t,scale*t, colors='black')



def SortRefTrans(raySet):
    refRays =   []
    transRays = [] 
    for i in range(len(raySet)):
        z = raySet[i].r[-1][-1]
        if abs(z) > thresh:
            transRays.append(raySet[i])  

        else:
            refRays.append(raySet[i])
    return [refRays,transRays]


#----- bin rays by 
def BinRaysAngle(raySet):
    # first separate all transmitted vs reflected rays -- use the z coordinate
    [refRays,transRays] = SortRefTrans(raySet)
    #now bin rays by angle and return
    refBin = []
    transBin =[]
    bins = [refBin,transBin]
    for i,rays in enumerate([refRays,transRays]):
        bin = bins[i]
        sign = 1
        if i==0: sign = -1
        for ray in rays:
            bin.append( vectorAngle(np.array([0,0,sign*1]), ray.k) )
    return bins


def BinRaysPosition(raySet):
    # first separate all transmitted vs reflected rays -- use the z coordinate
    [refRays,transRays] = SortRefTrans(raySet)
    #now bin rays by angle and return
    refBin = []
    transBin =[]
    bins = [refBin,transBin]
    for i,rays in enumerate([refRays,transRays]):
        bin = bins[i]
        for ray in rays:
            bin.append( np.linalg.norm(ray.r[-1][1:3]) )
    return bins


def BinRaysInteraction(raySet):
    # first separate all transmitted vs reflected rays -- use the z coordinate
    [refRays,transRays] = SortRefTrans(raySet)
    #now bin rays by angle and return
    refBin = []
    transBin =[]
    bins = [refBin,transBin]
    for i,rays in enumerate([refRays,transRays]):
        bin = bins[i]
        for ray in rays:
            bin.append( len(ray.r)-1 )
    return bins

def fibonacci_sphere(samples=1000):
    #https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    #note => there will be overlapp or underlapp
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points
def BinRaysBSDF(raySet,n):
    #step 1 get the rays and bins
    #rays
    [refRays,transRays] = SortRefTrans(raySet)
    #bins/k vectors
    trans = []
    ref = []
    pts = fibonacci_sphere(samples=n)
    for pt in np.array(pts):
        # ref
        if vectorAngle(np.array([0,0,-1]),pt) <= pi/2:
            ref.append(pt)
            trans.append(np.array(pt)*np.array([0,0,-1]))

    #step 2 get the cone angle
    s=1.0
    nr=len(ref)
    r = np.sqrt((1+s)*2/nr)
    theta = np.arctan(r)/2

    #now bin reflected rays
    binnedRefRays = []
    for binVec in ref:
        rayBin = []
        for ray in refRays:
            if vectorAngle(ray.k,binVec) <= theta:
                #append
                rayBin.append(ray)
        binnedRefRays.append([binVec,rayBin])

    #now bin reflected rays
    binnedTransRays = []
    for binVec in trans:
        rayBin = []
        for ray in transRays:
            if vectorAngle(ray.k,binVec) <= theta:
                #append
                rayBin.append(ray)
        binnedTransRays.append([binVec,rayBin])

    return [binnedRefRays,binnedTransRays]
    