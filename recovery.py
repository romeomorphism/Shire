from filters import FiltersGaussian
from filters import filter_in_space
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from skimage.feature import peak_local_max
import pandas as pd
import csv
from shire import ImageSeries

class Recovery(FiltersGaussian):
    def __init__(self, sigma=..., size=..., scale=..., n_directions=15) -> None:
        super().__init__(sigma, size, scale)
        self.filters = self.kernels_direction(n_directions)
        self.directions_grid = np.linspace(0, 2*np.pi, n_directions)
        self.return_unit()
        self.stage = 'Initialised'

    def return_unit(self):
        unit_x = 2*self.scale[0]*self.sigma[0] / (self.size[0]-1)
        unit_y = 2*self.scale[1]*self.sigma[1] / (self.size[1]-1)
        self.UNIT_PER_PIXEL = unit_x
        return (unit_x, unit_y)
    
    def compute_flow(self, images, start=0, frames = 100):
        self.start = start
        data = images.data[start:start+frames]

        self.flows = filter_in_space(matrix=data, filter=self.filters,  mode='reflect', multiple_filters=True)

        self.DISTANCE_PER_UNIT = images.DISTANCE_PER_PIXEL / self.UNIT_PER_PIXEL
        self.stage = 'Flow Computed'
        self.data = data
        self.dt = images.separation
    
    def estimate(self, thresholding, min_distance=3):

        max_flow_matrix = np.max(self.flows, axis=0)
        positions = []
        drt_index = []
        directions= []

        for i, flow in enumerate(max_flow_matrix):
            xy = peak_local_max(flow, threshold_abs=thresholding, min_distance=min_distance)
            xy_drt = []
            for cood in xy:
                xy_drt.append(np.argmax(self.flows[:,i,cood[0],cood[1]]))
            
            positions.append(xy)
            drt_index.append(xy_drt)
            directions.append(self.directions_grid[xy_drt])
        
        p_data = []
        d_data = []
        for _ in range(len(positions)):
            pos = positions[_]
            dir = directions[_]
            i = 0
            for p, d in zip(pos, dir):
                p_data.append({'Frame': _, 'Index': i,  'Position': p, 'X': p[0], 'Y':p[1]})
                d_data.append({'Frame': _, 'Index': i, 'Direction': d})
                i += 1
        
        self.df_p = pd.DataFrame(p_data)
        self.df_d = pd.DataFrame(d_data)

        self.stage = 'Direction Estimated'
        print("Direction Estimated, the info is stored in df_p and df_d")

    def plot(self, frame=0, path=None):
        pxy = np.array((self.df_p[self.df_p['Frame']==frame]['Position']).tolist())
        dxy = np.array((self.df_d[self.df_d['Frame']==frame]['Direction']).tolist())

        plt.scatter(pxy[:,1], pxy[:,0], c='r', s=15)
        plt.imshow(self.data[frame], cmap='gray', vmin=-1024, vmax=1024)

        for i, cood in enumerate(pxy):
            plt.arrow(cood[1], cood[0], \
                    25*np.cos(dxy[i])/3, \
                    -25*np.sin(dxy[i])/3, \
                        head_width=3, head_length=4, linewidth=3, color='aqua', length_includes_head=True)
            plt.text(cood[1], cood[0], f"{i}", fontsize=8, color='r')        
        plt.axis(False)
        plt.show()

    def label_particle(self, frame=0, plot=True):
        if plot:
            self.plot(frame)

        label = input("Enter the particle label for tracking: \n")
        self.particle_p = self.df_p[self.df_p['Frame']==frame]['Position'].iloc[int(label)]
        self.particle_d = self.df_d[self.df_d['Frame']==frame]['Direction'].iloc[int(label)]

        if plot:
            illustrate_particle(self.data[self.start+frame], self.particle_p, self.particle_d)

    def track_particle(self, speed_grid=np.linspace(0,0.1,201), num_frames=2, backward=False):

        p_list = [self.particle_p]
        d_list = [self.particle_d]
        s_list = [0.0]

        if not backward:
            for i in range(len(self.df_p['Frame'].unique())//self.dt-num_frames):
                print(f"Frame: {i*self.dt} / {len(self.df_p['Frame'].unique()) -self.dt*num_frames}")
                pxy = p_list[-1]
                dxy = d_list[-1]

                flows_speed = 0

                for k in range(num_frames):
                    filters = FiltersGaussian(self.sigma, self.size, self.scale).kernels_speed(dxy, speed_grid, (k+1)*self.dt)
                    flows_speed += filter_point(self.data[i*self.dt+(k+1)*self.dt], filters, pxy)
                s_list.append(speed_grid[np.argmax(flows_speed)])

                p_guess = pxy + np.array([-s_list[-1]* np.sin(dxy), s_list[-1]+np.cos(dxy)])
                index = find_nearest_position(p_guess, dxy, self.df_p[self.df_p['Frame'] == i*self.dt+self.dt]['Position'].tolist(), self.df_d[self.df_d['Frame'] == i*self.dt+self.dt]['Direction'].tolist())
                # index = find_nearest_position(p_guess, dxy, self.df_p[i*self.dt+self.dt], self.directions[i*self.dt+self.dt])
                p_list.append(self.df_p[self.df_p['Frame'] == i*self.dt+self.dt]['Position'].iloc[index])
                d_list.append(self.df_d[self.df_d['Frame'] == i*self.dt+self.dt]['Direction'].iloc[index])
                # p_list.append(self.positions[i*self.dt+self.dt][index])
                # d_list.append(self.directions[i*self.dt+self.dt][index])
        
        self.df_particle = pd.DataFrame({'Frame': range(self.start, self.start+self.dt*len(p_list), self.dt), 'Position': p_list, 'X': [p[0] for p in p_list], 'Y': [p[1] for p in p_list], "Direction": d_list, "Speed": s_list})
        return self.df_particle

    # @property
    # def df_particle(self):
    #     return pd.merge(self.df_particle_p, self.df_particle_d, on='Frame')
    
    def track_all(self, speed_grid=np.linspace(0,0.2,401), frame=0, num_frames=2, backward=False):
        self.df_all = pd.DataFrame()
        for i in range((self.df_p[self.df_p['Frame']==frame]).shape[0]):
            print(f"Particle: {i} / {(self.df_p[self.df_p['Frame']==frame]).shape[0]}")

            self.particle_p = self.df_p[self.df_p['Frame']==frame]['Position'].iloc[i]
            self.particle_d = self.df_d[self.df_d['Frame']==frame]['Direction'].iloc[i]
            df_particle = self.track_particle(speed_grid, num_frames, backward)

            df_particle['Particle'] = i
            self.df_all = pd.concat([self.df_all, df_particle])

    def mean_speed(self):
        return self.df_all.groupby('Particle')['Speed'].mean() * self.DISTANCE_PER_UNIT * ImageSeries.SCALE_CST
            
    def return_speeds(self):
        SCAlING_CST = self.DISTANCE_PER_UNIT * ImageSeries.SCALE_CST
        return  [s * SCAlING_CST for s in self.s_list]
    
    def return_directions(self):
        return self.d_list
    
    def return_positions(self):
        return self.p_list

    def return_travel_length(self):
        return 0.1 * self.dt *np.sum(self.s_list) * self.DISTANCE_PER_UNIT * ImageSeries.SCALE_CST
    
    def plot_path(self, particle_index=0, num_points=-1, labelled=False, save=False, path=None):
        positions = self.df_all[self.df_all['Particle']==particle_index]['Position'].tolist()
        illustrate_path(self.data[self.start], positions, num_points, labelled, save, path)



def plot_data_filter(ImagesSeries, DirectionRecovery, frame=0, filter_index=0, processed=True, save=False, path=None):

    if processed:
        img = ImagesSeries.data_processed[frame]
    else:
        img = ImagesSeries.data_raw[frame]

    plt.subplot(1,2,1)
    filter = DirectionRecovery.filters[filter_index]
    plt.imshow(np.pad(filter, (((img.shape[0] - filter.shape[0])//2, ), ((img.shape[1] - filter.shape[1])//2, ))), \
               cmap='gray', vmin=-1024, vmax=1024)
    plt.title(f"Filter: {filter_index}")

    plt.subplot(1,2,2)
    plt.imshow(img, cmap='gray', vmin=-1024, vmax=1024)
    plt.title(f"Processed: {processed}; Frame: {frame}")

    if save:
        plt.imsave(path, img)

def illustrate_particle(background, p, d):
    plt.imshow(background, cmap='gray', vmin=-1024, vmax=1024)
    plt.scatter(p[1], p[0], s=15)
    plt.arrow(p[1], p[0], \
                25*np.cos(d)/3, \
                -25*np.sin(d)/3, \
                    head_width=3, head_length=4, linewidth=3, color='yellow', length_includes_head=True)
    plt.axis(False)

def filter_point(matrix, filter, cood, mode='reflect'):
    if matrix.ndim == 2:
        return (filter_in_space(matrix[None,], filter, mode, multiple_filters=True)).squeeze()[:, *cood]
    else:
        return filter_in_space(matrix, filter, mode, multiple_filters=True)[:, *cood]
    
def find_nearest_position(p, d, positions, directions, thresholding=100):
    distances = np.zeros(len(positions))
    for i in range(len(positions)):
        # distances[i] = np.linalg.norm(p-positions[i])
        distances[i] = np.linalg.norm(p-positions[i]) + np.linalg.norm(d-directions[i])
    if np.min(distances) > thresholding:
        print("No particles in the range")
        return None
    else:
        return np.argmin(distances)
    
def illustrate_path(background, p_list, num_points=-1, labelled=False, save=False, path=None):
    fig, ax = plt.subplots()

    ax.imshow(background, cmap='gray', vmin=-1024, vmax=1024)
    # colors = cm.rainbow(np.linspace(0, 1, len(p_list)))
    # for i, pxy in enumerate(p_list):
    #     plt.scatter(pxy[1], pxy[0], s=15, color=colors[i])
    #     if labelled:
    #         plt.text(pxy[1], pxy[0], str(i), color='red')
    ax.set_axis_off()
    p_array = np.asarray(p_list)
    cax = ax.scatter(p_array[:num_points,1], p_array[:num_points,0], c=range(p_array[:num_points].shape[0]),cmap=plt.cm.get_cmap('RdYlBu_r'),s=0.5)
    # for i in range(p_array[:num_points].shape[0]):
    #     ax.plot(p_array[i,1], p_array[i,0])
    cbaxes = fig.add_axes([0.78, 0.62, 0.1, 0.02])
    cbar = fig.colorbar(cax,orientation='horizontal',shrink=0.5, cax=cbaxes)
    # plt.colorbar()
    cbar.set_ticks([])
    cbar.set_label('Time (ms)', c='white')
    
    if save:
        fig.savefig(path, bbox_inches='tight', pad_inches=0)

    # plt.axis(False)