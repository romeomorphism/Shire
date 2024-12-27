import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from filters import filter_in_time


class ImageSeries:

    SCALE_CST = 5.853118423645322
    DISTANCE_PER_PIXEL = 1000 / 6.3291

    def __init__(self, **kwargs) -> None:
        if "path" in kwargs:
            self.data = ImageSeries.read_tiff(kwargs["path"])
        if "data" in kwargs:
            self.data = kwargs["data"]

    def crop(self, xy, width, height, illustrate=True, inplace=False):
        """
        Crop the image series

        Params:
        ------
        xy - tuple of x and y coordinates
        width - width of the cropped image
        height - height of the cropped image

        Returns:
        ------
        a numpy matrix of size n_frames * height * width
        """
        x, y = xy
        cropped_images = ImageSeries(data=self.data[:, x : x + width, y : y + height])
        if illustrate:
            if x < 0:
                x = self.shape[1] + x
            if y < 0:
                y = self.shape[2] + y
            xy.reverse()
            crop_area = Rectangle(
                xy, width, height, linewidth=1, edgecolor="r", facecolor="none"
            )
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(self.head())
            ax[0].add_patch(crop_area)

            im = ax[1].imshow(
                cropped_images.head(),
                extent=[xy[0], xy[0] + width, xy[1] + height, xy[1]],
            )
            ax[1].set_title("Cropped Area")
            plt.colorbar(im, ax=ax[1])
            plt.show()
        return cropped_images

    def smooth(self, method="gaussian", sigma2=5, length=7, illustrate=True):
        """
        Smooth the image series

        Params  :
        ------
        method - method of smoothing
        length - length of the filter

        Returns:
        ------
        a numpy matrix of size n_frames * height * width
        """
        if method == "gaussian":
            filter = np.array([np.exp(-i**2/sigma2) for i in range(-length//2, length//2+1)])
            filter = filter / np.sum(filter)
        filtered_images = ImageSeries(data=filter_in_time(self.data, filter=filter))

        print("Smoothing done, the shape of the filtered images is", filtered_images.shape)

        if illustrate:
            filtered_images.head(illustrate=True)

        return filtered_images
    
    def remove_background(self, separation=1):
        data = self.data[separation:] - self.data[:-separation]
        
        bgrm_images = ImageSeries(data=data)
        bgrm_images.separation = separation
        
        return bgrm_images

    def head(self, illustrate=False):
        """
        Display the first frame of the image series

        Params:
        ------
        index - index of the frame
        """

        img = self.data[0]
        if illustrate:
            import matplotlib.pyplot as plt

            plt.imshow(img, cmap='gray', vmin=-1024, vmax=1024)
            plt.show()
        return img

    @property
    def shape(self):
        """
        Return the shape of the image series
        """
        return self.data.shape

    def __len__(self):
        pass

    @staticmethod
    def read_tiff(path):
        """
        Read the file and convert it into a numpy matrix

        Params:
        ------
        path - Path to the multipage-tiff file

        Returns:
        ------
        a numpy matrix of size n_frames * height * width
        """
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            images.append(np.array(img))
        return np.array(images)
