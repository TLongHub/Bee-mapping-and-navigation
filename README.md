# Bee-mapping-and-navigation
RESURG project exploring possible algorithms for how bees map and navigate their environments.

Firstly, I have devised a function that generates a field with random point charges and takes start and end points for a straight flight path. It then simulates an arthropod with a single electroreceptive hair flying along this path and computes the forces on that hair. By looking at the minima and maxima of this force function, we can determine likely coordinates for where a point charge may lie and so we plot the flight path and its perpendicular 'peak lines' to visualise these locations. 

The question now is - what is the best selection of flight paths to take to most accurately map the entire field?