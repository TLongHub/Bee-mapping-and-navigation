#TASK IS TO DEFINE A LOCAL MAXIMUM LOCATOR WITH A THRESHOLD FOR NATURAL NOISE/VARIANCE
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

data = [0, 0, 1, 0, 0, 0, 2, 3, 2, 1, 0, 0, 0, 0, 1, 0, 3, 5, 8, 11, 9, 6, 3, 0, 0]
max_locator(data, 1)