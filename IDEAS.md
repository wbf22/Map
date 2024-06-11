
# Image type (Mip)
- make a new image type similar to svg but simpler
    - just a bunch of lines that are single commands
- mip \<size x\> \<size y\> \<hex rgba background, otherwise transparent\>
    - creates a mip image (first line must be this)
- line 
    - \<x\> \<y\> \<x\> \<y\> \<hex rgba color\> \<thickness, default 1>
    - \<algebraic equation\> \<hex rgba color\> \<thickness, default 1> \<x or y\>-\<x or y\>
    - creates a line
- curve \<x\> \<y\> \<x\> \<y\>  \<control point x\> \<control point y\> \<hex rgba color\> \<thickness, default 1>
    - creates a curve
- fill \<x\> \<y\> \<hex rgba color\>
    - does a microsoft paint type operation to fill the area
- pixel 
    - \<x\> \<y\> \<hex rgba color\>
    - \<x, y, =, !=, \<, \>, %, /, *, -, +, rand\>
    - changes the color for a single pixel

if added on the same line or the next line with a ' ' or '\t' applies to 
- animate [\<x\> \<y\> \<x rotation\> \<y rotation\>  \<x scale\> \<y scale\> \<duration\> ...] \<repeat count \>
    - the list is keyframes which mip will interpolate between
    - if done on a fill operation, and the fill ran against lines, the lines are translated too

We'll want to make an editor for this if possible


# Defining views
- each view is divided into 1000 segmants called dp
- the root view divides the actual screen resolution into 1000 x 1000 dp and translates everything that way
- define a view 
    - dp location in parent 
    - pixel location in parent
    - use rectangle
        - rounded or unrounded
    - use Mip (our custom Image type) syntax to define the shape and appearance
- resize mode
    - resize with children when resized
    - keep children the same size (same # pixels)
- overflow
    - spill out children
        - for views that are more than 1000 dp, they spill out into the void allowing the view to either be scrolled or panned by clicking and dragging
    - resize to contain children
- scroll mode
    - scroll hotkeys
    - pan hotkeys
- set user starting scroll or pan location




# Scribbles

Cubic Bezier
B(t) = (1 - t)³ * P0 + 3(1 - t)² * t * P1 + 3(1 - t) * t² * P2 + t³ * P3

Quadratic Bezier
B(t) = (1 - t)² * P0 + 2(1 - t) * t * P1 + t² * P2


B(t) is the point on the curve at parameter t (where t ranges from 0 to 1)
P0, P1, P2, and P3 are the points defining the curve
P0 and P3 are the start and end points, respectively
P1 and P2 are the control points

By varying t from 0 to 1, you can generate all the points on the curve.


