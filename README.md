# Bee-mapping-and-navigation
RESURG project exploring possible algorithms for how bees map and navigate their environments.

Firstly, I have devised a function that generates a field with random point charges and takes start and end points for a straight flight path. It then simulates an arthropod with a single electroreceptive hair flying along this path and computes the forces on that hair. By looking at the minima and maxima of this force function, we can determine likely coordinates for where a point charge may lie and so we plot the flight path and its perpendicular 'peak lines' to visualise these locations. 

The question now is - what is the best selection of flight paths to take to most accurately map the entire field?

22/06/23
    The code has been edited to consider the absolute values of the forces only and to consider a minimum threshold to be characterised as a peak. This makes it easier to differentiate between a minimum and the gap between two maximums. We now only consider maximums of the absolute forces which corresponds only to the effects of point charges. Using two perpendicular paths followed by the same two paths with a rotation of some angle - not equal to a multiple of pi/2 - we can, in theory, pinpoint the locations of these charges and knock out any extra previously potential charge locations. 

    The current problem is the sensitivity of the signals. There is a balance between a low threshold for a peak to ensure all peaks are picked up, and a high enough threshold to ensure no unwanted noise is included. 