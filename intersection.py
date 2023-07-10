def euclidean_distance(x, y):
    """Calculates the Euclidean distance between points x and y in R^2."""
    z = (y[0] - x[0], y[1] - x[1])
    dist_sq = z[0]**2 + z[1]**2
    dist = dist_sq**0.5
    return dist

### Want a function that finds intersection point of two lines
def intersection(line1, line2):
    """Takes two lists of line points and finds the exact intersection point.
    If there is no intersection point or if they are the same line, we return None."""
    #First find equations of lines
        #LINE ONE
    if (line1[-1][0] - line1[0][0]) == 0: #if line 1 vertical
        if (line2[-1][0] - line2[0][0]) == 0: #if both lines vertical
            return None #even if lines are the same...
        else:
            x = line1[0][1] #equation for line is x = a (vertical line)
    else:        
        m1 = (line1[-1][1] - line1[0][1]) / (line1[-1][0] - line1[0][0])
        c1 = -line1[0][0]*m1 + line1[0][1]
        #LINE TWO
    if (line2[-1][0] - line2[0][0]) == 0: #if line 2 vertical, but line 1 is not
        x = line2[0][0] #so input x from vertical line 2 (constant x)
        y = x*m1 + c1 #and get y from intersection with line 1
        return (x, y)
    else:
        m2 = (line2[-1][1] - line2[0][1]) / (line2[-1][0] - line2[0][0])
        c2 = -line2[0][0]*m2 + line2[0][1]
        if x == line1[0][1]: #if line 1 vertical, but line 2 not
            y = x*m2 + c2
            return (x, y)
    #if neither lines vertical
    #Find the intersection location
    if m1-m2 == 0:
        return None #Lines parallel - no intersection
    x = (c2-c1) / (m1-m2)
    #Input back into one of the equations
    y = m1*x + c1
    #Return the point of intersection
    return (x, y)

intersection([(0,1), (10,1)], [(0,0), (0,10)])


### Want a function that checks if a point is close enough
    #  to a line to be considered to lie on the line

def inclusion(point, line, tol):
    """Takes a point, a line and a tolerance value and returns True if
    the point is within tol distance from the line and False otherwise."""
    a = point[0]
    b = point[1]
    #For vertical line:
    if (line[-1][0] - line[0][0]) == 0: #if line vertical
        if a-tol <= line[0][0] <= a+tol: #if x lies in range of a
            return True #then point is sufficiently close
        else:
            return False
    #For non-vertical line:
    else:
        m = (line[-1][1] - line[0][1]) / (line[-1][0] - line[0][0])
        c = -line[0][0]*m + line[0][1] #find equation of line
        for x in [a-tol + 2*tol*i/1000 for i in range(1001)]: #x interval
            if (x-a)**2 + (m*x+c-b)**2 <= tol**2: #if line enters the circle 
                return True
    return False

inclusion((0,0), [(-2, 0.1), (2, 0.1)], 0.1)