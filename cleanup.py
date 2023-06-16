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
    length = euclidean_distance(start, stop) #The length of the path
    path_progress = [0 + i*(length/increments) for i in range(increments +1)]
    
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
    inverse = numpy.linalg.inv(transformation) #The inverse matrix of the transformation matrix
        #Next, we apply the matrix
    new_charges = {}
    for charge in charges: #Need to put charges into position vectors
        charge_vector = numpy.array([charge[0], charge[1], 0]) #Old coordinates charge vector
        tran_ch_vec = (numpy.matmul(transformation, charge_vector)) #The transformed charge vector in new coordinates
        new_charges[(tran_ch_vec[0], tran_ch_vec[1])] = charges[charge] #New dictionary with updated coordinates

        #Finally, we can compute the downward forces acting on the hair
    forces = [down_force(new_charges, x) for x in path_progress] #A list of the force on the hair at each point along the path

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
        for x in path_progress:
            if down_force(new_charges, x) == peak:
                peaks[x] = peak
    
    #Find turning point for path (distance D before last peak)
    turn_point = max(peaks) - D

    return path, path_progress, forces, peaks, turn_point, new_charges, gradient


#Now we can test the functions so far

charges = generate_charges(1, False)
path, path_progress, forces, peaks, turn_point, new_charges = fly_path((0, 0), (10, 0), charges, 0.5)

peak_x = [peak for peak in peaks]
peak_val = [peaks[peak] for peak in peaks]


#Now we want to determine polarity with the first path
    #If we first fly along the edge of our field, we know that every charge will
    #  be on one side of us, thus we can determine their polarity

#Assume a 10x10 field
def find_polarities(N, charges, plot = False):
    """Determine the polarities of charges from an initial flight path."""
    start = (0, 0)
    stop = (10, 0)
    path, path_progress, forces, peaks, turn_point, new_charges, gradient = fly_path(start, stop, charges, 1)
    polarities = []
    for x in peaks:
        if peaks[x] > 0:
            polarities.append(True)
        else:
            polarities.append(False)

    #Plotting the force/displacement graph
    if plot == True:
        peak_x = [peak for peak in peaks]
        peak_val = [peaks[peak] for peak in peaks]

        plt.figure(figsize=(10,10))
        plt.plot(path_progress, forces)
        for i in range(len(peak_x)):
            if polarities[i] == True: #If charge positive
                plt.plot(peak_x, peak_val, 'o', color = 'r')
            elif polarities[i] == False: #If charge negative
                plt.plot(peak_x, peak_val, 'o', color = 'g')
        plt.vlines(turn_point, min(forces), max(forces), 'b', 'dashed', label = turn_point)
        plt.xlim(0, 10)
        plt.legend()
        plt.show()

    return path, path_progress, forces, peaks, turn_point, new_charges, gradient, polarities

find_polarities(1, charges)

#Let's put the plotting into functions
    #Plotting the forces graph
def plot_forces(forces, path_progress, peaks, turn_point):

    peak_x = [peak for peak in peaks] #Find the peak coordinates
    peak_val = [peaks[peak] for peak in peaks]

    plt.figure(figsize=(10,10))
    plt.plot(path_progress, forces)
    plt.plot(peak_x, peak_val, 'o')
    plt.vlines(turn_point, min(forces), max(forces), 'g', 'dashed', label = turn_point)
    plt.xlim(0, 10)
    plt.legend()
    plt.show()

plot_forces(forces, path_progress, peaks, turn_point)

    #Plotting the actual and possible charge locations
def plot_field(charges, peaks):
    N = len(charges) #Number of point charges
    x = []   # Find locations
    y = []
    mag = []
    for charge in charges:
        x.append(charge[0])
        y.append(charge[1])
        mag.append(charges[charge])   # Find charges

    plt.figure(figsize=(10,10)) #Plot 10x10 figure
    ax = plt.gca()
    plt.plot(x, y, 'o') #Plot the point charges
    for i in range(N):
        ax.annotate(mag[i], xy=(x[i]+0.1, y[i]+0.1), xytext=(x[i]+1, y[i]+1), #Annotate the charges with magnitude
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    )
    
    peak_x = [peak for peak in peaks] #Find the peak coordinates
    plt.vlines(peak_x, 0, 10) ### NEEDS WORK: This is where we plot 
                        # the lines to show potential charge locations
    
    plt.xlim(0, 10) #Restrict to 10x10 plot
    plt.ylim(0, 10)
    plt.show()

plot_field(charges, peaks)

#Need a way of storing the data found so far: 
# possible coordinates of charges, polarities, and magnitudes.
def data_single():
    #For each path flown, we store the data found in a list
    # key: path_num, value: [[path info], [peak info], [charge info]]
    data = {}
    charges = generate_charges(1)
    path, path_progress, forces, peaks, turn_point, new_charges, gradient, polarities = find_polarities(1, charges)
    min_max = [] #Find if peaks are min or max
    peak_val = [] #Find peak values
    peak_x = [] #Find peak locations
    for peak in peaks: 
        if peaks[peak] > 0: #Max
            min_max.append(True)
        elif peaks[peak] < 0: #Min
            min_max.append(False)
        peak_val.append(peaks[peak]) #Values
        peak_x.append(peak) #Locations along path

    peak_xy = []

    data[1] = [[(0,0), (10,0)], [min_max, peak_val, peak_x]]
