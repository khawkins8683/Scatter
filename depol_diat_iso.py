import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import time

import utility as u
import material as m
import ray as r

import monte_carlo as mc
import plotting as plot




def main():
    #step 1 ---- RECREATE the plots from the paper
    #figure 1 - Cos[theta]
    # set Ratio   
    ratio = 10   
    thick = 70E-6  
    n = 1000
    air = m.Material2D(1.0,1000000,  thick)
    glass = m.Material2D(1.5 + 1E-7*1j, thick/ratio, thick)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        rays = [r.Ray(u.normalize([0,0,1]),  np.array([0,0,0.1]), 0.5 ) for _ in range(n)]
        args = [[ray, air,glass, False] for ray in rays]
        print('running: ratio: ',ratio)
        start = time.time()
        raysOut = [val for val in executor.map( mc.MonteCarloTrace, args )]
        end = time.time()
        print('timing: ',end - start,' (s) ',len(raysOut),' per ray ',(end - start)/len(raysOut))
    
    
    print('sorting')
    start = time.time()
    [refRays,_] = plot.SortRefTrans(raysOut)
    end = time.time()
    print('timing: ',end - start,' (s) ref rays: ',len(refRays),' ratio ', len(refRays)/len(raysOut) )

    print('binning')
    start = time.time()
    
    dataBin = plot.binRays2DScatNKH(raysOut, np.array([0,0,-1]), 20,  1  )
    end = time.time()
    print('timing: ',end - start,' (s) bins: ',len(dataBin))


    # here we will calculate depol instead of number of rays
    print('plotting:  ')
    xCosBin = [];yCosBin = []
    for d in dataBin:
        xCosBin.append(d[0])#input angle
        mmList = [ray.MM[1] for ray in d[1]]#[ray.MMCumulative() for ray in d[1] ]
        mm = mmList[0]#np.sum(mmList,axis=0)#now sum mm
        #print("Test: mm ",mm,mm[0,0])
        yCosBin.append(mm[0,0])

    xCosBin2 = [];yCosBin2 = []
    for d in dataBin:
        xCosBin2.append(d[0])#input angle
        mmList = [ray.MM[-1] for ray in d[1]]#[ray.MMCumulative() for ray in d[1] ]
        mm = mmList[0]#np.sum(mmList,axis=0)#now sum mm
        #print("Test: mm ",mm,mm[0,0])
        yCosBin2.append(mm[0,0])        

    xCosBin3 = [];yCosBin3 = []
    for d in dataBin:
        xCosBin3.append(d[0])#input angle
        mmList = [ray.MMCumulative() for ray in d[1]]#[ray.MMCumulative() for ray in d[1] ]
        mm = mmList[0]#np.sum(mmList,axis=0)#now sum mm
        #print("Test: mm ",mm,mm[0,0])
        #print("\n\nMM   ",mm,d[1][0].MM)
        yCosBin3.append(mm[0,0])  

    
    #yCosFit = np.max(yCosBin)*np.cos(xCosBin)
    #plt.plot(x,y,marker = 'o')
    #plt.show()
    fig, ax = plt.subplots(1,3,figsize=(15, 15))#, figsize=(15, 5)
    ax[0].plot(xCosBin,yCosBin,marker='o')
    #ax[0].plot(xCosBin,yCosFit,marker='o')
    ax[0].set_xlabel('Scatter Angle [rad]')
    ax[0].set_ylabel('Intensity (# mm Sum /bin)')
    ax[0].set_title('Intensity Profile Input')


    ax[1].plot(xCosBin2,yCosBin2,marker='o')
    ax[1].plot(xCosBin3,yCosBin3,marker='o')
    #ax[0].plot(xCosBin,yCosFit,marker='o')
    ax[1].set_xlabel('Scatter Angle [rad]')
    ax[1].set_ylabel('Intensity (# mm Sum /bin)')
    ax[1].set_title('Intensity Profile Output')


    ax[2].plot(xCosBin3,yCosBin3,marker='o')
    #ax[0].plot(xCosBin,yCosFit,marker='o')
    ax[2].set_xlabel('Scatter Angle [rad]')
    ax[2].set_ylabel('Intensity (# mm Sum /bin)')
    ax[2].set_title('Intensity Profile Path')

    # n=1000
    # thick = 70E-6
    # ratios = [0.1,0.5,1,2,5,10,15,20,25,30]
    # xRef = [];    yRef=[]
    # for ratio in ratios:

    #     air = m.Material2D(1.0,1000000,  thick)
    #     glass = m.Material2D(1.5 + 1E-7*1j, thick/ratio, thick)

    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         rays = [r.Ray(u.normalize([0,0,1]),  np.array([0,0,0.1]), 0.5 ) for _ in range(n)]
    #         args = [[ray, air,glass, False] for ray in rays]
    #         print('running: ratio: ',ratio)
    #         start = time.time()
    #         raysOut = [val for val in executor.map( mc.MonteCarloTrace, args )]
    #         end = time.time()
    #         print('timing: ',end - start,' (s) ',len(raysOut),' per ray ',(end - start)/len(raysOut))

    #     print('sorting')
    #     start = time.time()
    #     [refRays,transRays] = plot.SortRefTrans(raysOut)
    #     end = time.time()
    #     print('timing: ',end - start,' (s) ref rays: ',len(refRays),' ratio ', len(refRays)/len(raysOut) )
    #     xRef.append(ratio)
    #     yRef.append(len(refRays)/len(raysOut) )




    # ax[1].plot(xRef,yRef,marker='o')
    # ax[1].set_xlabel('Thickness Ratio [thickness / mean travel distance]')
    # ax[1].set_ylabel('Reflectivity')
    # ax[1].set_title('Reflectivity (# Reflected rays / # Total input rays)')
    plt.show()
    #plot reflectivitiy vs ( thick/mean distance )

    # y = np.array(test)[:,0]
    # plt.hist(y,20)
    # 
    # print(test[0])
    # print(glass)


    #step 2 --- diattenuation + depol vs theta scatter

    #step 3 --- on axis diattenuation  
    
    

if __name__ == '__main__':
    main()