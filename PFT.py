"""
Created on Sat Apr  3 20:20:34 2021

@author: Deepayan Banik
"""
import matplotlib.pyplot as plt
import numpy as np

dx = 0.005
dy = 0.005
x = np.arange(-1, 1, dx)
y = np.arange(-1, 1, dy)
x1 = np.linspace(-1, 1, len(x)-1)
y1 = np.linspace(-1, 1, len(y)-1)
  
# Creating 2-D grid of features
[X, Y] = np.meshgrid(x,y)

''' Strengths of different elements '''
U = 1.0 # uniform flow
m = 1.0 # source strength
gamma = 1.0 # vortex strength
kappa = 1.0 # doublet strength
R = 0.5 # radius of impactor shank
b = m/(2*np.pi*U) # rankine oval parameter

phi = 0

def impactor(xs,ys): # from Sharma 2019
    return (R**2 * X / np.sqrt((Y - ys)**2 + (X - xs)**2) - 2 * (Y - ys)**2) * U / 4

def source(xs,ys):
    return m / (2 * np.pi) * np.arctan2((Y - ys) , (X - xs))
    
def vortex(xs,ys):
    return gamma / (2 * np.pi) * np.log(np.sqrt((Y - ys)**2 + (X - xs)**2))

def doublet(xs,ys):
    return - kappa / (2 * np.pi) * np.sin(np.arctan2((Y - ys) , (X - xs))) / np.sqrt((Y - ys)**2 + (X - xs)**2)

''' Flow description '''

phi = phi + U * Y
#phi = phi + doublet(0.0,0.0)
#phi = phi + vortex(0.0,0.0)
#phi = phi + impactor(0.0,0.0)

nsources = 10
rad_dist_sources = 0.2
theta =np.linspace(0.001,360.0-360.0/nsources,nsources)
for th in theta:
    ''' sources on a circle '''
#    phi = phi + source(rad_dist_sources * np.cos(th*np.pi/180),rad_dist_sources * np.sin(th*np.pi/180)) 
    ''' sources on a rankine oval '''
    phi = phi + source(b*(np.pi-th*np.pi/180)/np.tan(th*np.pi/180),b*(np.pi-th*np.pi/180))
    

''' Calculating velocity fields '''
phit = np.transpose(phi)
u = np.zeros((len(phi)-1,len(phi)-1)) 
v = np.zeros((len(phi)-1,len(phi)-1)) 
vmag = np.zeros((len(phi)-1,len(phi)-1)) 
for i in range(0,len(phi)-1):
    u[i] = np.diff(phit[i])/dx
    v[i] = -np.diff(phi[i])/dy

''' Processing high gradients or outliers, often due to sources '''
umean = np.mean(u) # mean
ustd = np.std(u) # standard deviation
vmean = np.mean(v)
vstd = np.std(v)
print(str(ustd) + "\t" + str(vstd) + "\t" + str(max(map(max,u))) + "\t" + str(max(map(max,v))))
##
nstd=2 # multiple of standard deviation allowed in the domain
for i in range(0,len(phi)-1): # replacing outliers with averages
    for j in range(0,len(phi)-1):
        if (abs(u[i][j]-umean) > nstd*ustd):
            u[i][j]=(u[i][j-1]+u[i][j+1])/2
        if (abs(v[i][j]-vmean) > nstd*vstd):
            v[i][j]=(v[i][j-1]+v[i][j+1])/2
for i in range(0,len(phi)-1):
    for j in range(0,len(phi)-1):
        vmag[i][j]=np.sqrt(u[j][i]**2+v[i][j]**2) # total magnitude of velocity
#%%            
plt.rcParams['contour.negative_linestyle'] = 'solid'
fig = plt.figure(figsize=(17,5))
plt.subplot(1, 4, 1)

''' plots streamlines '''
plt.contour(X, Y, phi, 200, colors='black')
plt.title('Streamlines')
#plt.contourf(X, Y, phi, 100)

c=4
plt.subplot(1, 4, 2)
''' plots u velocity '''
plt.imshow(u.T,cmap ='magma',vmin=-c,vmax=c,interpolation ='nearest',origin ='lower')
plt.title('u-velocity')
plt.colorbar(orientation='horizontal')

plt.subplot(1, 4, 3)
''' plots v velocity '''
plt.imshow(v,cmap ='magma',vmin=-c,vmax=c,interpolation ='nearest',origin ='lower')
plt.title('v-velocity')
plt.colorbar(orientation='horizontal')

plt.subplot(1, 4, 4)
''' plots velocity magnitude '''
plt.imshow(vmag,cmap ='binary',vmin=0,vmax=2*c,interpolation ='nearest',origin ='lower')
plt.title('velocity magnitude')
plt.colorbar(orientation='horizontal')
