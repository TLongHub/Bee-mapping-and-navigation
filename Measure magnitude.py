# AIM: to create an algorithm to measure magnitude of charges
import math, numpy, random, matplotlib.pyplot as plt


#FIRST WE COPY IN USEFUL FUNCTIONS

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
    if start[0] != stop[0]:
        gradient = (stop[1]-start[1])/(stop[0]-start[0]) #How much y increase happens for unit x increase
        theta = math.atan(gradient) #The angle between x-axis and path
    else:
        theta = numpy.pi/2 #To get around division by zero problem
    h = 0 - start[0] #Difference in x from old origin to new origin
    k = 0 - start[1] #The same for y
        #Now, we formulate a transformation matrix
    transformation = numpy.array([[math.cos(theta), -math.sin(theta), h*math.cos(theta) - k*math.sin(theta)],
                                   [math.sin(theta), math.cos(theta), h*math.sin(theta) + k*math.cos(theta)],
                                    [0, 0, 1]]) #Transformation matrix
    inverse = numpy.linalg.inv(transformation) #The inverse matrix of the transformation matrix
        #Next, we apply the matrix
    new_charges = {}
    for charge in charges: #Need to put charges into position vectors
        charge_vector = numpy.array([charge[0], charge[1], 1]) #Old coordinates charge vector
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

    #Need to apply inverse matrix to find the positions in the original set up
    # and the peak lines
    peak_lines = []
    for x in peaks: #Form the peak lines in transformed set up
        peak_lines.append([numpy.array([x, -10, 1]), numpy.array([x, 10, 1])])
    
    og_peak_lines = []
    for line in peak_lines: #Apply inverse matrix to get 'original' peak lines
        og_peak_lines.append([numpy.matmul(inverse, line[0]), numpy.matmul(inverse, line[1])])
    og_peak_pos = []
    for x in peaks: #Apply inverse matrix to get 'original' peak locations along path
        og_peak_pos.append(numpy.matmul(inverse, numpy.array([x, 0, 1])))
    

    return path, path_progress, forces, turn_point, peaks, og_peak_pos, peak_lines, og_peak_lines

#COPY COMPLETE

charges = generate_charges(1, plot = True)
B = 2
path_1, path_progress_1, forces_1, turn_point, peaks_1, og_peak_pos_1, peak_lines, og_peak_lines = fly_path((0,0), (10,0), charges, 0.5)
path_2, path_progress_2, forces_2, turn_point, peaks_2, og_peak_pos_2, peak_lines, og_peak_lines = fly_path((0,B), (10,B), charges, 0.5)

maxima_mag_1 = []
for peak in peaks_1:
    maxima_mag_1.append(peaks_1[peak])

maxima_mag_2 = []
for peak in peaks_2:
    maxima_mag_2.append(peaks_2[peak])

print(maxima_mag_1, maxima_mag_2)

radii = []
for i in range(len(maxima_mag_1)):
    r = maxima_mag_1[i]/maxima_mag_2[i]
    R = (-B)/(r-1)
    radii.append(R)

#...not exactly working well...

