import math, numpy, random, matplotlib.pyplot as plt


def down_force(charges, bee_location):
    """Assuming moving along x-axis only for now."""
    total_force = 0
    for charge in charges:
        total_force += (charges[charge]*charge[1])/((bee_location-charge[0])**2 + charge[1]**2)**(5/2)
    return total_force

def generate_charges(N, plot = True):
    """Creates a dictionary of N point charge locations and their magnitudes,
    and visually plots them."""
    charges = {}
    for _ in range(N):
        charges[(random.random()*10, random.random()*10)] = (random.random()-0.5)*4
        
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
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.show()   # Plot charge locations and charges
    
    return charges



def fly_path(start, stop, charges, D):
    """Fly the path described and measure the force on the hair. 
    Notes the locations and sign of peaks and location of turning point (x1 + D)."""
    increments = 1000 #The number of increments
    dx = (stop[0]-start[0])/increments #Change in x
    dy = (stop[1]-start[1])/increments #Change in y
    path = [(start[0] + dx*i, start[1] + dy*i) for i in range(increments)] #The path flown
    
    #Now we need to measure the downward force on the hair assuming the path is the x-axis
        #First, we need to translate and rotate the current points in space (charges)
        # to fit the new axis
        #So we find the angle of rotation (angle between x-axis and path)
    gradient = (stop[1]-start[1])/(stop[0]-start[0]) #How much y increase happens for unit x increase
    theta = math.atan(gradient) #The angle between x-axis and path
    h = start[0] #Difference in x from old origin to new origin
    k = start[1] #The same for y
        #Now, we formulate a transformation matrix
    transformation = numpy.array([[math.cos(theta), -math.sin(theta), h*math.cos(theta) - k*math.sin(theta)],
                                   [math.sin(theta), math.cos(theta), h*math.sin(theta) + k*math.cos(theta)],
                                    [0, 0, 1]]) #Transformation matrix
        #Next, we apply the matrix
    new_charges = {}
    for charge in charges: #Need to put charges into position vectors
        charge_vector = numpy.array([charge[0], charge[1], 0]) #Old coordinates charge vector
        tran_ch_vec = numpy.matmul(transformation, charge_vector) #The transformed charge vector in new coordinates
        new_charges[tran_ch_vec] = charges[charge] #New dictionary with updated coordinates
    
        #Finally, we can compute the downward forces acting on the hair
    forces = [down_force(charges, x) for x in path] #A list of the force on the hair at each point along the path
    
    #We now need to find all the nim/max points
    l = forces #Copy forces
    l = [x for x,y in zip(l[0:], l[1:]) if x != y] + [l[-1]] #Remove identical neighbors
    l = [0] + l + [0] #Append [0] to both endpoints
    # Retain elements where each of their neighbors are greater than them
    local_min = [y for x, y, z in zip(l[0:], l[1:], l[2:]) if x > y < z] #All the local mins
    local_max = [y for x, y, z in zip(l[0:], l[1:], l[2:]) if x < y > z] #All the local maxs
    peak_vals = local_min + local_max

    #Note the x-value for the location of the mins/maxs
    peaks = {} #X-value (location) and peak value (prop to magnitude but not polarity)
    for peak in peak_vals:
        for x in path:
            if down_force(charges, x) == peak:
                peaks[x] = peak
    
    #Find turning point for path (distance D before last peak)
    turn_point = max(peaks) - D

    return path, forces, peaks, turn_point
