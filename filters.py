import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import multivariate_normal

def filter_in_time(matrix, filter=[1], interval=1):
    """
    Filter a 3D matrix in time.

    Parameters:
        matrix (np.array): 3D matrix to be filtered.
        filter (list): Filter to be applied to the matrix.
        interval (int): Interval between each filter application.
        illustration (bool): If True, plot the filtered matrix.
    """
    filtered_matrix = np.zeros(((matrix.shape[0]-len(filter)) // interval + 1, *matrix.shape[1:]))
    for i in range(filtered_matrix.shape[0]):
        filtered_matrix[i] = np.average(matrix[i * interval:i * interval + len(filter),:,:], axis=0, weights=filter)
    return filtered_matrix

def filter_in_space(matrix, filter, mode='reflect', multiple_filters=False):
    """
    Filtering process is conducted by corelation in the code
    """
    
    if multiple_filters:
        return np.array([np.array([ndi.correlate(image, kernel, mode=mode) for image in matrix]) \
                     for kernel in filter])
    else:
        return np.array([ndi.correlate(image, filter, mode=mode) for image in matrix])

class FiltersGaussian:
    def __init__(self, sigma=(0.1,0.1), size=(5,5), scale=(5,5)) -> None:
        super().__init__()
        self.sigma = sigma
        self.size = size
        self.scale= scale

    def return_unit(self):
        unit_x = 2*self.scale[0]*self.sigma[0] / (self.size[0]-1)
        unit_y = 2*self.scale[1]*self.sigma[1] / (self.size[1]-1)
        return (unit_x, unit_y)
    
    def kernels_direction(self, num_directions=15):
        directions = np.linspace(0, 2*np.pi, num_directions)
        return np.array([dgauss2d_kernel(sigma=self.sigma, scale=self.scale, angle=dirt, size=self.size) for dirt in directions])
    
    def kernels_speed(self, angle, speed_grid, dt):
        return np.array([dgauss2d_kernel(sigma=self.sigma, scale=self.scale, angle=angle, size=self.size, v=speed, dt=dt) for speed in speed_grid])
    


def gauss2d(x, y, mu=[0,0], sigma=[1,1], order=[0,0]):
    x, y = np.meshgrid(x, y)
    xy = np.dstack((x, y))

    normal_rv = multivariate_normal(mu, sigma)
    z = normal_rv.pdf(xy)

    w = len(x)
    h = len(y)
    title="2D-Gauss"
    if order == [1,0]:
        z = -(x-mu[0])/sigma[0]**2*z.reshape(w, h, order='F')
        title="y-derivative of 2D-Gauss"
    elif order == [0,1]:
        z = -(y-mu[1])/sigma[1]**2*z.reshape(w, h, order='F')
        title="x-derivative of 2D-Gauss"
    elif order == [0,2]:
        z = (y**2/sigma[1]**4-1/sigma[1]**2)*z.reshape(w, h, order='F')
        title="Second order x-derivative of 2D-Gauss"
    elif order == [2,0]:
        z = (x**2/sigma[1]**4-1/sigma[1]**2)*z.reshape(w, h, order='F')
        title="Second order y-derivative of 2D-Gauss"
    elif order == [2,2]:
        z = ((x**2+y**2)/sigma[1]**4-2/sigma[1]**2)*z.reshape(w, h, order='F')
        title="Second order y-derivative of 2D-Gauss" 

    return np.flipud(z)

def dgauss2d_kernel(sigma=(0.1,0.1), scale=(3,3), angle=0, size=(10, 10), v=0, dt=0, normalized=True):
    x = np.linspace(-scale[0]*sigma[0], scale[0]*sigma[0], size[0])
    y = np.linspace(-scale[1]*sigma[1], scale[1]*sigma[1], size[1])


    mu = [v*dt*np.cos(angle), v*dt*np.sin(angle)]
    z = np.cos(angle) * gauss2d(x, y, mu, sigma, order=[1,0]) + np.sin(angle) * gauss2d(x, y, mu, sigma, order=[0,1])
    kernel = -z
    if normalized:
        kernel = kernel * sigma[0]**4
    return kernel