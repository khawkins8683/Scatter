import functools 

import matplotlib.pyplot as plt
import numpy as np
import pySCATMECH as scatmech
from pySCATMECH.fresnel import *

thresh = 1E-10
deg = np.pi/180
stack = scatmech.fresnel.FilmStack()
glass = scatmech.fresnel.OpticalFunction(1.5)
air = scatmech.fresnel.OpticalFunction(1)

def normalize(x):
    norm = np.divide(x ,np.linalg.norm(x) )
    return norm

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

def IsotropicPhaseFunction(N):
    theta = np.random.rand(N) * np.pi
    phi = np.random.rand(N) * 2*np.pi
    return [theta, phi]

def CreateSP(eta,k):
    #  TODO special case for normal incidence
    if sum(abs(np.subtract(eta,k))) > thresh:
        s = normalize(np.cross(k,eta))
    else:
        s = normalize(np.cross(k, np.add([0,-1,0],k) ))
    p = np.cross(k,s)
    return np.array([s,p,k])


def KVector(theta,phi):
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ])

def RotationMatrix(u,v):
    if sum(abs(np.subtract(u,v))) > thresh:
        a = np.cross(u,v)
        [ax,ay,az] = normalize(a)
        theta = np.arcsin( np.linalg.norm(a)  )
        Sa = np.array([[0,-1*az,ay],[az,0,-1*ax],[-1*ay,ax,0]])
        a = np.array([ax,ay,az])[np.newaxis]
        A = np.dot(a.T,a)
        R = A +(np.identity(3) - A)*np.cos(theta) + Sa*np.sin(theta)
    else:
        R = np.identity(3)
    return R

def KScatter(kIn,theta,phi):
    RotZtoKin = RotationMatrix([0,0,1],kIn)
    kNewZ = KVector(theta,phi)
    kScat = np.dot(RotZtoKin, kNewZ)
    return kScat.transpose().flatten()


def RayleighJones(thetaScat): # I don't think we need phi scat just yet
    jm = np.array([[np.cos(thetaScat),0],[0,1]])
    return jm

def JonesAddDim(jm):
    jm3d = np.array([
        [jm[0][0], jm[0][1], 0 ],
        [jm[1][0], jm[1][1], 0 ],
        [0,0,1]
    ]) 
    return jm3d

def PRT(jm,eta,kin,kScatter):
    oIn= CreateSP(eta,kin)
    oOut = CreateSP(eta,kScatter)
    prt = functools.reduce( np.dot, [oOut.transpose() , JonesAddDim(jm), oIn ])
    return prt

#for fresnel scatter -> eta is midway between the input K and scatter K
def TIRCheck(n1,n2,eta,kin):   
    mu = (n1/n2)
    sIn = np.cross(eta,kin)
    g1 = ((mu**2)*np.linalg.norm(sIn)**2)
    #if g1>1 then we have TIR and need to run reflect3D
    if (g1>1):
        return True
    else:
        return False
#Todo snell and reflect 3D
def Refract3D(n1,n2,eta,kin, debug= False):   
    mu = (n1/n2)
    sIn = np.cross(eta,kin)
    metaXk = np.cross(eta,kin)#np.cross(-1*eta,kin)

    factor1 = mu*np.cross(eta,metaXk)
    g1 = ((mu**2)*np.linalg.norm(sIn)**2)
    #if g1>1 then we have TIR and need to run reflect3D
    if (g1>1):
        if debug: print("TIR")
        return Reflect3D(eta,kin)
    factor2 = eta*np.sqrt(1-g1)
    kOut = -1*(factor1-factor2)
    return kOut

def Reflect3D(eta,kIn):
        #Reflect k across eta
        kRef = kIn - 2*(kIn.dot(eta))*eta
        return kRef


def vectorAngle(v1, v2):
    v1n = normalize(v1)
    v2n = normalize(v2)
    return np.arccos(np.clip(np.dot(v1n, v2n), -1.0, 1.0))

def SignedVectorAngle(v1,v2,sign = 1):
    angle = vectorAngle(v1, v2)
    cross = np.cross(v1,v2)
    if np.array([0,0,sign*1]).dot( cross )<0:
        angle = -1*angle

    return angle

def SnellsLaw(n1,n2,theta):
    return np.arcsin( (n1/n2)*np.sin(theta) )




## now we need to do a 'ray' trace
## set up material thickness
## set up material properties
## TODO - add debug printing
## TODO - add in prt - jones from fresnel

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
                k = Refract3D(DiffuseMaterial.n, 1 , np.array([0,0,-1]), RayPath.k )
                live = TIRCheck(DiffuseMaterial.n,1.0,np.array([0,0,-1]),RayPath.k)
            if z>thickness:
                if debug: print("Transmit")
                deltaZ = RayPath.r[-1][-1] - thickness #-thickness for transmission
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

        #now update ray parameters
        RayPath.k = k
        RayPath.r = np.append(RayPath.r , np.array([r]), axis = 0)
        if i > 1000:
            live = False
        
    return RayPath

def MonteCarloTrial(N,Mat,wavelength,kIn, debug= False):
    if debug: i=0
    rayList = []
    for i in range(N):
        if debug: i+=1
        if debug: print( "Tracing Ray i: ", i )
        rayList.append( MonteCarloTrace( Ray(kIn, np.array([0,0,0]), wavelength ), Mat, debug  ) )
    return rayList



## -------------------------------------------------------------------------------------------------------------------------------------
##Classes
## -------------------------------------------------------------------------------------------------------------------------------------


class Material:
    def __init__(self, n, gamma, thickness):
        self.n = np.real(n)
        self.k = np.imag(n)
        self.gamma = gamma
        self.thickness = thickness


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
        if z > 0:
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