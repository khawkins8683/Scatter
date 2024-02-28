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

    n=1000
    air = m.Material2D(1.0,1000000,70E-6)
    glass = m.Material2D(1.5 + 1E-7*1j, 5E-6, 70E-6)
    #print(air)
    #print(glass)


    with concurrent.futures.ProcessPoolExecutor() as executor:
        #results = [executor.submit(testF) for ii in range(10)]
        
        rays = [r.Ray(u.normalize([0,0,1]),  np.array([0,0,0.1]), 0.5 ) for _ in range(n)]
        args = [[ray, air,glass, False] for ray in rays]


        start = time.time()
        raysOut = [val for val in executor.map( mc.MonteCarloTrace, args )]
        end = time.time()
        print('timing: ',end - start,' (s) ',len(raysOut),' per ray ',(end - start)/len(raysOut))

    print('sorting')
    start = time.time()
    [refRays,transRays] = plot.SortRefTrans(raysOut)
    end = time.time()
    print('timing: ',end - start,' (s) ref rays: ',len(refRays),' ratio ', len(refRays)/len(raysOut) )

    print('binning')
    start = time.time()
    dataBin = plot.binRays2DScatNKH(raysOut, np.array([0,0,-1]), 20,  1  )
    end = time.time()
    print('timing: ',end - start,' (s) bins: ',len(dataBin))


    #step 1 ---- RECREATE the plots from the paper
    print('plotting:  ')
    x = []
    y = []
    for d in dataBin:
        x.append(d[0])
        if d[0] == 0.0:
            y.append(2*len(d[1])  )#we need to double the 0 bin => weird
        else:
            y.append(len( d[1])  )

    plt.plot(x,y,marker = 'o')
    y2 = np.max(y)*np.cos(x)
    plt.plot(x,y2)
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