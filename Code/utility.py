import functools 
import numpy as np


thresh = 1E-15
deg = np.pi/180

#Simple linear algebra
def normalize(x):
    norm = np.divide(x ,np.linalg.norm(x) )
    return norm

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

#General Optics
def SnellsLaw(n1,n2,theta):
    return np.arcsin( (n1/n2)*np.sin(theta) )

# KVector stuff
def KVector(theta,phi):
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta) ])


def KAnglesThetaPhi(kVector, sign=1):
    theta = np.arccos(kVector.dot(np.array([0,0,sign*1]) ))
    phi = np.arctan(kVector[1]/kVector[0])
    return [theta,phi]


def KScatter(kIn,theta,phi):
    RotZtoKin = RotationMatrix([0,0,1],kIn)
    kNewZ = KVector(theta,phi)
    kScat = np.dot(RotZtoKin, kNewZ)
    return kScat.transpose().flatten()

## Ray trace functions
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

def Refract3D(n1,n2,eta,kin, debug= False):   
    mu = (n1/n2)
    sIn = np.cross(eta,kin)
    metaXk = np.cross(eta,kin)#np.cross(-1*eta,kin)
    g1 = ((mu**2)*np.linalg.norm(sIn)**2)
    #if g1>1 then we have TIR and need to run reflect3D
    if (g1>1):
        if debug: print("TIR")
        return Reflect3D(eta,kin)
    factor1 = mu*np.cross(eta,metaXk)
    factor2 = eta*np.sqrt(1-g1)
    kOut = -1*(factor1-factor2)
    return kOut

def Reflect3D(eta,kIn):
        #Reflect k across eta
        kRef = kIn - 2*(kIn.dot(eta))*eta
        return kRef

##  Matrix
def CreateSP(eta,k):
    #  TODO special case for normal incidence
    if sum(abs(np.subtract(eta,k))) > thresh:
        s = normalize(np.cross(k,eta))
    else:
        s = normalize(np.cross(k, np.add([0,-1,0],k) ))
    p = np.cross(k,s)
    return np.array([s,p,k])


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


## PRT --------------------
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
