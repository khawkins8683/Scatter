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
    thick = 70E-6
    ratios = [0.1,0.5,1,2,5,10,15,20,25,30]
    reflectivity = []
    for ratio in ratios:

        air = m.Material2D(1.0,1000000,  thick)
        glass = m.Material2D(1.5 + 1E-7*1j, thick/ratio, thick)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            #results = [executor.submit(testF) for ii in range(10)]
            
            rays = [r.Ray(u.normalize([0,0,1]),  np.array([0,0,0.1]), 0.5 ) for _ in range(n)]
            args = [[ray, air,glass, False] for ray in rays]

            print('running: ratio: ',ratio)
            start = time.time()
            raysOut = [val for val in executor.map( mc.MonteCarloTrace, args )]
            end = time.time()
            print('timing: ',end - start,' (s) ',len(raysOut),' per ray ',(end - start)/len(raysOut))

        print('sorting')
        start = time.time()
        [refRays,transRays] = plot.SortRefTrans(raysOut)
        end = time.time()
        print('timing: ',end - start,' (s) ref rays: ',len(refRays),' ratio ', len(refRays)/len(raysOut) )
        reflectivity.append([ratio, len(refRays)/len(raysOut) ])



    #step 1 ---- RECREATE the plots from the paper
    print('plotting:  ')
    x =np.array(reflectivity)[:,0]
    y =np.array(reflectivity)[:,1]
    plt.plot(x,y,marker = 'o')
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