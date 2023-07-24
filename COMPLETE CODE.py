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

def generate_charges(N, delta, plot = True):
    """Creates a dictionary of N point charge locations and their magnitudes,
    and visually plots them."""
    global D1
    min_mag = 2
    num_charges = 0 
    attempts = 0

    while num_charges < N: #Keep trying until we get N charges
        #Every 50 attempts, we reset and try again
        if attempts % 100 == 0:
            charges = {}
            num_charges = 0

        #GENERATE charge properties
            #Magnitude
        magnitude = random.random()*5 + min_mag #(between 2 and 7)
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

        #ACT on results of checks
        attempts += 1
        if acceptable == True:
            charges[(x,y)] = magnitude
            num_charges += 1
    
    if plot == True:
        plot_field(charges, N)
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
    global D1
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

    peak_vals = max_locator(forces, 0.01) #THRESHOLD - tolerance value can be changed as needed
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


# NEXT WE FLY THIRD PATH, CONVERT CANDIDATE POINTS INTO NEW COORDINATE SYSTEM
#  AND SEE WHICH ARE VALID

def validate_candidates(start, stop, candidates, charges):
    
    trans_mat, length, path_peak_locations = fly_path(start, stop, charges) #Fly angled 3rd path
    print("\nThe number of 3rd path peaks is:", len(path_peak_locations))
    inverse_trans_mat = numpy.linalg.inv(trans_mat) #Converts into new c.s.

    new_cs_candidates = [] #Convert candidates into new coordinate system
    for candidate in candidates:
        new_cs_candidates.append(numpy.matmul(inverse_trans_mat, candidate))

    charge_location_estimates = []
    for candidate in new_cs_candidates:
        for location in path_peak_locations:
            if abs(candidate[0] - location) <= 0.25: #Check candidate lies on new c.s. line
                charge_location_estimates.append(numpy.matmul(trans_mat, candidate))
    
    return charge_location_estimates


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
    for location in estimates:
        x.append(location[0])
        y.append(location[1])
    plt.plot(x, y, 'x')     

    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.show()   # Plot charge locations and charges


    #Generate charges
#N = 2
#charges = generate_charges(N, 1.5, plot = False)
    #Find candidates (xy c.s.)
#candidates = find_candidates((0,2.5), (5,2.5), (2.5,0), (2.5,5), charges)
#print("The number of candidates are:", len(candidates))

    #Initiate angled 3rd path
#start = (0,0)
#stop = (5,5)
#charge_location_estimates = validate_candidates(start, stop, candidates, charges)

#print(charge_location_estimates, charges, "\nThe number of estimates are:", len(charge_location_estimates))

    #Plot findings
#plot_estimates(candidates, charge_location_estimates, charges, N)



#VALIDATING WITH TWO COORDINATES

def validate_candidates_v2(start1, stop1, start2, stop2, candidates, charges):
        #FOR NOW, FIX THE PATHS
    start1 = (0,0)
    stop1 = (5,5)
    start2 = (5,0)
    stop2 = (0,5)    
    #Transform charges into new coordinate system
    #Fly horizontal and vertical paths
    trans_mat, length1, path_peak_locations1 = fly_path(start1, stop1, charges)
    _, length2, path_peak_locations2 = fly_path(start2, stop2, charges)
    #Mark x' and y' coordinates
    x_prime = path_peak_locations1
    y_prime = [coord - length2/2 for coord in path_peak_locations2] 
            # '- length2/2' to acount for paths intersecting in their middle
    #Find new candidates in this c.s.
    candidates_prime = []
    for x in x_prime:
        for y in y_prime:
            candidates_prime.append((x,y))
    #Convert old candidates into this c.s.
    inverse_trans_mat = numpy.linalg.inv(trans_mat) #Converts into new c.s.
    new_cs_candidates = [] #Convert candidates into new coordinate system
    for candidate in candidates:
        new_cs_candidates.append(numpy.matmul(inverse_trans_mat, candidate))
    #Compare candidates and return matches
    estimates = []
    for point in new_cs_candidates: #Working in new angled c.s.
        for point2 in candidates_prime:
            if euclidean_distance(point, point2) < 1: #If sufficiently close
                estimates.append(numpy.matmul(trans_mat, point)) #Add the original c.s. candidate to estimates
  
    return estimates

D1 = 8 #Arbitrarily selected
    #Generate charges
N = 8
charges = generate_charges(N, 0, plot = False) #2nd arg is delta (min distance between charges on each axis)
    #Find candidates (xy c.s.)
candidates = find_candidates((0,2.5), (5,2.5), (2.5,0), (2.5,5), charges)
print("The number of candidates are:", len(candidates))

estimates = validate_candidates_v2(0, 0, 0, 0, candidates, charges)
plot_estimates(candidates, estimates, charges, N)
print(len(estimates))



### TESTING ###
# Firstly, what proportion of runs yield the same number of estimates as actual charges?
N = 2
def find_prop_success():
    tally = 0
    for _ in range(1000):
        charges = generate_charges(N, 0.25, plot = False)
        candidates = find_candidates((0,2.5), (5,2.5), (2.5,0), (2.5,5), charges)
        estimates = validate_candidates_v2(0, 0, 0, 0, candidates, charges)
        if len(estimates) == len(charges):
            tally += 1
    print("The proportion of runs that guess the right number of charges is:", tally/1000)

#find_prop_success()

#Secondly, what proportion of 'successful' runs (those which estimate the right number of charges)
#  have all estimates within distance x of actual charge location? 
def find_prop_closer_than_x(x):
    successes = 0
    tally = 0
    while successes < 1000:
        charges = generate_charges(N, 0.25, plot = False)
        candidates = find_candidates((0,2.5), (5,2.5), (2.5,0), (2.5,5), charges)
        estimates = validate_candidates_v2(0, 0, 0, 0, candidates, charges)
        if len(estimates) == len(charges):
            successes += 1
            accurate = True
            for estimate in estimates:
                for charge in charges:
                    if x < euclidean_distance(estimate, charge) < 1.5: #Close enough to be the right estimate but too far to be accurate
                        accurate = False
            if accurate == True:
                tally += 1
    print("The proportion of successful runs that estimate with an accuracy of within distance x from actual location are:", tally/successes)

x = 0.2
#find_prop_closer_than_x(x)

#Next, RMSE of estimates from actual charges
def find_RMSE(delta):
    successes = 0
    tally = 0
    SE = []
    while successes < 1000:
        charges = generate_charges(N, delta, plot = False)
        candidates = find_candidates((0,2.5), (5,2.5), (2.5,0), (2.5,5), charges)
        estimates = validate_candidates_v2(0, 0, 0, 0, candidates, charges)
        if len(estimates) == len(charges):
            successes += 1
            for estimate in estimates:
                for charge in charges:
                    if euclidean_distance(estimate, charge) < 1: #Close enough to be the right estimate but too far to be accurate
                        SE.append(euclidean_distance(charge, estimate)**2)
    MSE = sum(SE)/len(SE)
    RMSE = MSE**0.5
    print("The RMSE for 2000 estimated charge locations is:", RMSE)

