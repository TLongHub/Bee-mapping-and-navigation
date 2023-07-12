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
    min_mag = 2 #Used to simplify scenario
    min_dist = 2.5 #Hopefully makes the peaks more distinct
    charges = {}
    for _ in range(N):
        
        magnitude = 0 
        while abs(magnitude) < min_mag: #If magnitude not large enough, generate a new one
            magnitude = (random.random()-0.5)*10

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
        if keep_location == True:
            charges[(x,y)] = magnitude
        
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


def max_locator(data, threshold):
    """Takes a list of data and a threshold and finds all the local maxima whose
    absolute value are greater than that threshold. Returns these data points."""
    maxima = [] #Initialise the maxima list
    
    if data[0] > data[1] and abs(data[0]) > threshold: #Checking if first point is a local maxima
        maxima.append(data[0]) #Appending that point        # and is over the threshold
    
    for i in range(1, len(data)-1):
        if data[i-1] < data[i] > data[i+1] and abs(data[i]) > threshold: 
            maxima.append(data[i]) #Check maxima and over threshold and append

    if data[-1] > data[-2] and abs(data[-1]) > threshold:
        maxima.append(data[-1]) #The same for last point
    
    return maxima

def axis_path_angle(start, stop):
    if stop[0] != start[0]:
        gradient = (stop[1]-start[1])/(stop[0]-start[0]) #How much y increase happens for unit x increase
    
    if stop[1] > start[1]: #Path going 'up'
        if stop[0] > start[0]:
            theta = math.atan(gradient)
        elif stop[0] == start[0]:
            theta = numpy.pi/2 #Avoid division by zero
        else:
            theta = math.atan(gradient) + numpy.pi

    elif stop[1] == start[1]: #Path running flat (horizontal)
        if stop[0] < start[0]:
            theta = numpy.pi #Ensure correct direction
        else:
            theta = 0

    else: #Path going 'down'
        if stop[0] > start[0]:
            theta = math.atan(gradient)
        elif stop[0] == start[0]:
            theta = 3*numpy.pi/2 #270 degree rotation to ensure correct direction
        else:
            theta = math.atan(gradient) + numpy.pi #Plus 180 to ensure direction correct
    
    return theta


def fly_path(start, stop, charges):
    """Fly the path described and measure the force on the hair. 
    Notes the locations and sign of peaks and location of turning point (x1 + D)."""

    #INITIALISING VARIABLES 
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
    
    theta = axis_path_angle(start, stop)
        
        #And then the translation of the origin

    h = 0 - start[0] #Difference in x from old origin to new origin
    k = 0 - start[1] #The same for y

        #Now, we formulate a transformation matrix and its inverse

    transformation = numpy.array([[math.cos(-theta), -math.sin(-theta), h*math.cos(-theta) - k*math.sin(-theta)],
                                   [math.sin(-theta), math.cos(-theta), h*math.sin(-theta) + k*math.cos(-theta)],
                                    [0, 0, 1]]) #Transformation matrix
    inverse = numpy.linalg.inv(transformation) #The inverse matrix of the transformation matrix
        
        #Next, we apply the matrix to get the charge locations relative to the path

    new_charges = {}
    for charge in charges: #Need to put charges into position vectors
        charge_vector = numpy.array([charge[0], charge[1], 1]) #Old coordinates charge vector
        tran_ch_vec = (numpy.matmul(transformation, charge_vector)) #The transformed charge vector in new coordinates
        new_charges[(tran_ch_vec[0], tran_ch_vec[1])] = charges[charge] #New dictionary with updated coordinates

        #Finally, we can compute the downward forces acting on the hair
            #(forces acting from left to right of path)
    
    forces = [abs(down_force(new_charges, x)) for x in path_progress] #A list of the absolute value of the 
                                                            # forces on the hair at each point along the path
        #Find the peak locations

    peak_vals = max_locator(forces, 0.001) #tolerance value can be changed as needed
    path_peak_locations = [x for x in path_progress if (abs(down_force(new_charges, x)) in peak_vals)] #the locations of peaks

        #Find the peak locations in the xy coordinate system in the field
    
    xy_peak_locations = []
    for x in path_peak_locations: #Apply inverse matrix to get xy field peak locations
        location = numpy.matmul(inverse, numpy.array([x, 0, 1]))
        xy_peak_locations.append(location)
    
    return inverse, length, path_peak_locations
        #returns list of arrays with coordinates of the peak locations 
            #Note: direction of 'peak line' implied by direction of path (perp)


def find_coordinates(start, stop, charges):
    """Find x and y coordinates of candidate locations for charges."""
    if start[0] == stop [0]: #Check if path is vert or horiz
        vertical = True
    else:
        vertical = None

    if start[1] == stop[1]:
        horizontal = True
    else: 
        horizontal = None

    inverse, length, path_peak_locations = fly_path(start, stop, charges) #Fly the path

    xy_peak_locations = []
    for x in path_peak_locations: #Apply inverse matrix to get xy field peak locations
        location = numpy.matmul(inverse, numpy.array([x, 0, 1]))
        xy_peak_locations.append(location)

    xi = []
    for point in xy_peak_locations: #Find the x or y coordinates xi/yi
        if vertical == True:
            xi.append(point[1])
        elif horizontal == True:
            xi.append(point[0])
        else:
            print("Neither horizontal nor vertical path.")
            return None
        
    return xi, vertical, horizontal


def find_candidates(start1, stop1, start2, stop2, charges):
    """Find coordinates of candidates for locations of charges."""
    ai, vertical1, horizontal1 = find_coordinates(start1, stop1, charges)
    bi, vertical2, horizontal2 = find_coordinates(start2, stop2, charges)

    if horizontal1 == True: #Ensure we have correct x and y coordinates
        xi = ai
        yi = bi
    else:
        xi = bi
        yi = ai

    candidates = [] 
    for x in xi: #Combine coordinates to get every possible candidate point
        for y in yi:
            candidates.append(numpy.array([x, y, 1]))

    return candidates

charges = generate_charges(2, plot = False)
candidates = find_candidates((0,2.5), (5,2.5), (2.5,0), (2.5,5), charges)
print("The number of candidates are:", len(candidates))


# NEXT WE FLY THIRD PATH, CONVERT CANDIDATE POINTS INTO NEW COORDINATE SYSTEM
#  AND SEE WHICH ARE VALID

def validate_candidates(start, stop, candidates, charges):
    
    trans_mat, length, path_peak_locations = fly_path(start, stop, charges) #Fly angled 3rd path
    inverse_trans_mat = numpy.linalg.inv(trans_mat) #Converts into new c.s.

    new_cs_candidates = [] #Convert candidates into new coordinate system
    for candidate in candidates:
        new_cs_candidates.append(numpy.matmul(inverse_trans_mat, candidate))

    charge_location_estimates = []
    for candidate in new_cs_candidates:
        for location in path_peak_locations:
            if abs(candidate[0] - location) <= 0.4: #Check candidate lies on new c.s. line
                charge_location_estimates.append(numpy.matmul(trans_mat, candidate))
    
    return charge_location_estimates

start = (0,0)
stop = (5,5)
charge_location_estimates = validate_candidates(start, stop, candidates, charges)

print(charge_location_estimates, charges, "\nThe number of estimates are:", len(charge_location_estimates))



    #Plot charges and estimate locations
def plot_estimates(candidates, estimates, charges, N):
    #Plot charges (blue)
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
    #Plot candidates (red)
    x = []
    y = []
    for candidate in candidates:
        x.append(candidate[0])
        y.append(candidate[1])
    plt.plot(x, y, 'x') 

    #Plot estimates (green)
    x = []
    y = []
    for location in charge_location_estimates:
        x.append(location[0])
        y.append(location[1])
    plt.plot(x, y, 'x')     

    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.show()   # Plot charge locations and charges

plot_estimates(candidates, charge_location_estimates, charges, 2)