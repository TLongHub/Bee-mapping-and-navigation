import math, numpy, random, matplotlib.pyplot as plt

def euclidean_distance(x, y):
    """Calculates the Euclidean distance between points x and y in R^2."""
    z = (y[0] - x[0], y[1] - x[1])
    dist_sq = z[0]**2 + z[1]**2
    dist = dist_sq**0.5
    return dist

def down_force(charges, bee_location):
    """Assuming moving along x-axis only for now."""
    total_force = 0
    for charge in charges:
        total_force += (charges[charge]*charge[1])/((bee_location-charge[0])**2 + charge[1]**2)**(5/2)
    return total_force

def generate_charges(N, delta, plot = True):
    """Creates a dictionary of N point charge locations and their magnitudes,
    and visually plots them."""
    global D1
    min_mag = 1 #Used to simplify scenario
    min_dist = 1 #Hopefully makes the peaks more distinct
    charges = {} 
    num_charges = 0 
    attempts = 0
    while num_charges < N:
        #MAGNITUDE
        magnitude = 0 
        while abs(magnitude) < min_mag: #If magnitude not large enough, generate a new one
            magnitude = (random.random())*10 #Positive magnitude only

        #LOCATION
        keep_location = False
        while keep_location == False:
            x = random.random()*5 #Generate new locations until we keep it
            y = random.random()*5
            if not bool(charges) == True: #If there are no other locations, we keep
                keep_location = True
            else:
                keep_location = True
                for charge in charges: #If there are others, we check they are far enough away
                    if euclidean_distance(charge, (x, y)) < min_dist:
                        keep_location = False
                    if abs(charge[0]-x) < delta or abs(charge[1]-y) < delta: #Min distance for x and y coordinates
                        keep_location = False
                    a = delta*numpy.sin(numpy.pi/4)
                    c1 = charge[0] + charge[1] - 2*a
                    c2 = charge[0] + charge[1] + 2*a
                    if c1 < y+x < c2: #Min diagonal distance of delta
                        keep_location = False
                    k1 = -charge[0] + charge[1] + 2*a
                    k2 = -charge[0] + charge[1] - 2*a
                    if k1 > y-x > k2: #Min diagonal distance of delta
                        keep_location = False
                    interaction = abs(magnitude*charges[charge]) / (euclidean_distance(charge, (x,y))**2)
                    if interaction > D1 / 4:
                        keep_location = False
        if keep_location == True:
            charges[(x,y)] = magnitude
            num_charges += 1
        
        #RESET IF STUCK IN INFINITE LOOP
        if keep_location == False:
            attempts += 1
            if attempts > 100:
                charges = {}
                num_charges = 0
                attempts = 0
        

    #PLOTTING   
    if plot == True:
        x = []   # Find locations
        y = []
        mag = []
        for charge in charges:
            x.append(charge[0])
            y.append(charge[1])
            mag.append(charges[charge])   # Find charges

        fig, ax = plt.subplots()
        plt.plot(x, y, 'o')
        for i in range(N):
            ax.annotate(mag[i], xy=(x[i]+0.1, y[i]+0.1), xytext=(x[i]+1, y[i]+1), 
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        )
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.show()   # Plot charge locations and charges
    
    return charges


#N = 2
D1 = 1.5
#generate_charges(N, 1)


def plot_field(charges, N):
    #PLOTTING   
    x = []   # Find locations
    y = []
    mag = []
    for charge in charges:
        x.append(charge[0])
        y.append(charge[1])
        mag.append(charges[charge])   # Find charges

    fig, ax = plt.subplots()
    plt.plot(x, y, 'o')
    for i in range(N):
        ax.annotate(mag[i], xy=(x[i]+0.1, y[i]+0.1), xytext=(x[i]+1, y[i]+1), 
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    )
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.show()   # Plot charge locations and charges


### NEW GENERATE CHARGES FUNCTION ###

def generate_charges_v2(N, delta, plot = True):
    """Creates a dictionary of N point charge locations and their magnitudes,
    and visually plots them."""
    global D1
    global MIN_MAG
    num_charges = 0 
    attempts = 0

    while num_charges < N: #Keep trying until we get N charges
        #Every 50 attempts, we reset and try again
        if attempts % 100 == 0:
            charges = {}
            num_charges = 0

        #GENERATE charge properties
            #Magnitude
        magnitude = random.random()*5 + MIN_MAG #(between 2 and 7)
            #Location
        x = random.random()*5 #(randomly placed in 5x5 field)
        y = random.random()*5
        
        #CHECK if properties acceptable
        acceptable = True
        for charge in charges:
            #Min distance
            if euclidean_distance(charge, (x, y)) < delta:
                acceptable = False
            #Min distance for x and y coordinates
            if abs(charge[0]-x) < delta or abs(charge[1]-y) < delta:
                acceptable = False
            #Min diagonal distance of delta
            a = delta*numpy.sin(numpy.pi/4)
            c1 = charge[0] + charge[1] - 2*a
            c2 = charge[0] + charge[1] + 2*a
            if c1 < y+x < c2: 
                acceptable = False
            k1 = -charge[0] + charge[1] + 2*a
            k2 = -charge[0] + charge[1] - 2*a
            if k1 > y-x > k2: 
                acceptable = False
            #Max interactive forces between charges
            interaction = abs(magnitude*charges[charge]) / (euclidean_distance(charge, (x,y))**2)
            if interaction > D1 / 4:
                acceptable = False
            else:
                print(interaction)

        #ACT on results of checks
        attempts += 1
        if acceptable == True:
            charges[(x,y)] = magnitude
            num_charges += 1
    
    if plot == True:
        plot_field(charges, N)
    return charges

MIN_MAG = 0.5
D1 = MIN_MAG / ((5*2**0.5)/2)**2
I_min = MIN_MAG*D1/4
charges = generate_charges_v2(2, 0.25, plot = True)
print(charges)
print(I_min, D1/4, D1)