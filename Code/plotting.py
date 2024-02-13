import math
import numpy as np
import matplotlib.pyplot as plt

from . import utility as u

#sorting/utility--------------------------------------------------------

def SortRefTrans(raySet):
    refRays =   []
    transRays = [] 
    for i in range(len(raySet)):
        z = raySet[i].r[-1][-1]
        if abs(z) > u.thresh:
            transRays.append(raySet[i])  

        else:
            refRays.append(raySet[i])
    return [refRays,transRays]

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


#Path Plots-----------------------------------------------------------
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

#3D
def PlotRayPath3D(Ray,ax):
    positions = Ray.r
    #x values are z
    xVals = positions[:,0]
    yVals = positions[:,1]
    zVals = positions[:,2]
    ax.plot(xVals,yVals,zVals,  marker='o')


#Histograms-----------------------------------------------------------------------------
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
            bin.append( u.vectorAngle(np.array([0,0,sign*1]), ray.k) )
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
            bin.append( np.linalg.norm(ray.r[-1][0:2]) )
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


#BSDF --------------------------
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
        if u.vectorAngle(np.array([0,0,-1]),pt) <= np.pi/2:
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
            if u.vectorAngle(ray.k,binVec) <= theta:
                #append
                rayBin.append(ray)
        binnedRefRays.append([binVec,rayBin])

    #now bin reflected rays
    binnedTransRays = []
    for binVec in trans:
        rayBin = []
        for ray in transRays:
            if u.vectorAngle(ray.k,binVec) <= theta:
                #append
                rayBin.append(ray)
        binnedTransRays.append([binVec,rayBin])

    return [binnedRefRays,binnedTransRays]