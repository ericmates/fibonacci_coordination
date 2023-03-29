from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import math

from ase.io import read, write
from ase.data import vdw_radii
from ase import Atoms

def fibonacci_sphere(radius, samples=1000):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return points

atoms = read('example.cif') # molecular structure (any format)
molecule = atoms[0, 1, 2, 32, 38, 56, 57, 58] # atom indices of atoms where sphere is to be generated
p_dens = .25 # atoms per A squared
min_height = 10. # minimum distance at which points will not be considered. This is relevant for structures in which we are not interested in placing atoms below.
elements_to_avoid = 'O' # if any coordinate is too close from this element, they will be removed.

_x, _y, _z = [], [], [] 
for atom in molecule:
    radius = vdw_radii[atom.number] + 1
    a, b, c = atom.position
    area = 4.*np.pi*radius**2.
    points = int( area * p_dens)
    
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 20)
    x = np.outer(np.sin(theta), np.cos(phi))*radius + atom.x
    y = np.outer(np.sin(theta), np.sin(phi))*radius + atom.y
    z = np.outer(np.cos(theta), np.ones_like(phi))*radius + atom.z

    coords = fibonacci_sphere(radius, samples=points)
    xi, yi, zi = np.array(coords)[:,0]*radius+atom.x, np.array(coords)[:,1]*radius+atom.y, np.array(coords)[:,2]*radius+atom.z
    _x.extend(xi)
    _y.extend(yi)
    _z.extend(zi)

for atom in molecule:   # removes the points that are inside any sphere, effectively keeping only coordinates that are on the vdw radius of the molecule
    radius = vdw_radii[atom.number]
    a, b, c = atom.position
    j=1
    for i in range(0, len(_x)):
        if ((_x[-j]-a)**2 + (_y[-j]-b)**2 + (_z[-j]-c)**2)**.5 < radius-.1:
            _x.remove(_x[-j])
            _y.remove(_y[-j])
            _z.remove(_z[-j])
        else:
            j+=1

# for i in range(0, len(_x)):   # remove points that are below a given distance
#     if _z[i] < min_distance:
#         _x.remove(_x[i])
#         _y.remove(_y[i])
#         _z.remove(_z[i])

conds = np.ones_like(_x, dtype=bool)
for atom in molecule:
    conds = np.where(((atom.position[0]-_x)**2 + (atom.position[1]-_y)**2 + (atom.position[2]-_z)**2)**.5 < vdw_radii[atom.number]-.2, False, conds)
conds = np.where(_z < np.full_like(conds, min_height, dtype=float), False, conds)
conds = np.where(_x < np.full_like(conds, 9.2, dtype=float), False, conds) # same as trimming per height, but for the x axis (we are not interested in adsorbing from the left)
positions_avoid = [atom.position for atom in atoms if atom.symbol == elements_to_avoid]

for position in positions_avoid:
    conds = np.where(((position[0]-_x)**2 + (position[1]-_y)**2 + (position[2]-_z)**2)**.5 < 1.5, False, conds)

X = np.array(_x)[conds]
Y = np.array(_y)[conds]
Z = np.array(_z)[conds]

for i in range(0, len(X)):
    h = Atoms('H', positions=[(X[i], Y[i], Z[i])])
    structure = atoms + h
    write(f'structure{i}.xyz', structure)
    
h = Atoms('He', positions=[(X[0], Y[0], Z[0])])
for i in range(1, len(X)):
    h2 = Atoms('He', positions=[(X[i], Y[i], Z[i])])
    h = h + h2
write('structure_with_all_atoms_as_he.xyz', atoms + h)
