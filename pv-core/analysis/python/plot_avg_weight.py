"""
Make a histogram of normally distributed random numbers and plot the
analytic PDF over it
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import PVReadWeights as rw

"""
Show how to modify the coordinate formatter to report the image "z"
value of the nearest pixel given x and y
"""

if len(sys.argv) < 2:
   print "usage: plot_avg_weight filename"
   sys.exit()

#space = 1

weights = rw.PVReadWeights(sys.argv[1])

nx = weights.nx
ny = weights.ny
nxp = weights.nxp
nyp = weights.nyp

def format_coord(x, y):
   col = int(x+0.5)
   row = int(y+0.5)
   numcols = nx
   numrows = ny
    
   if col>=0 and col<numcols and row>=0 and row<numrows:
      z = P[row,col]
      return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
   else:
       return 'x=%1.4f, y=%1.4f'%(x, y)

#nx_im = nx * (nxp + space) + space
#ny_im = ny * (nyp + space) + space

wavg = np.zeros([nx * ny])

for i in range(len(wavg)):
   nextp = weights.next_patch()
   ap = np.average(nextp)
   wavg[i] = ap

wavg = wavg.reshape(nx, ny)
tavg = np.average(wavg)

print "total average weight = ", tavg
print nx, '&', ny

#im = np.zeros((nx_im, ny_im))
#im[:,:] = (weights.max - weights.min) / 2.

#for k in range(weights.numPatches):
#   P = weights.next_patch()
#   if len(P) != nxp * nyp:
#      continue

#   P = np.reshape(P, (nxp, nyp))
#   numrows, numcols = P.shape

#   x = space + (space + nxp) * (k % nx)
#   y = space + (space + nyp) * (k / nx)

#   im[y:y+nyp, x:x+nxp] = P

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlabel('Kx GLOBAL')
ax.set_ylabel('Ky GLOBAL')
ax.set_title('Weight Patches')
ax.format_coord = format_coord

ax.imshow(wavg, cmap=cm.jet, interpolation='nearest', vmin=weights.min, vmax=weights.max)

plt.show()

#end fig loop
