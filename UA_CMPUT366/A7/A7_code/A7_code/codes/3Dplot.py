import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data = np.load('values.npy')
x = np.repeat(np.arange(50)[None,:],50,axis=0)
y = np.repeat(np.arange(50)[:,None],50,axis=1)
ax.plot_surface(x,y,-data.T,cstride=1,rstride=1)

ax.set_ylabel('Velocity')
ax.set_xlabel('Position')
ax.set_zlabel('Value')
ax.set_title('The Cost-to-Go Function Learned During One Run.')
plt.show()
