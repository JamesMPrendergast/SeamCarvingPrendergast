import numpy as np
import cv2
from typing import List, Tuple
from KinkaidDecorators import log_start_stop_method

class Carver:
    energy_image: np.ndarray
    cumulative_energy_image: np.ndarray

    def __init__(self):
        pass

    def calculate_energy(self, source_image) -> np.ndarray:
        self.energy_image = cv2.Sobel(source_image.astype(float),
                                    ddepth=cv2.CV_16S,
                                    dx=1,
                                    dy=0,
                                    ksize=3,
                                    borderType=cv2.BORDER_REFLECT)
        self.energy_image = (np.abs(self.energy_image)).astype(np.uint8)
        self.energy_image = cv2.cvtColor(self.energy_image, cv2.COLOR_BGR2GRAY)
        return self.energy_image

    # @log_start_stop_method
    def generate_cumulative_energy_grid(self, energy_image) -> np.ndarray:
        """
        Based on the information in self.energy_image, constructs the cumulative grid
        :return: the cumulative grid that goes with this energy grid.
        """
        # start the cumulative grid off as a grid of zeros, the same size as the energy grid, with a copy of the top row
        #    of energy in its top row. So if "energy" is
        #    1 2 3 4
        #    5 6 7 8
        #    9 1 2 3
        #    4 5 6 7
        # then "cumulative" _starts_off_ as
        #    1 2 3 4
        #    0 0 0 0
        #    0 0 0 0
        #    0 0 0 0

        self.cumulative_energy_image = np.zeros(energy_image.shape, dtype=float)
        self.cumulative_energy_image[0, :] = energy_image[0, :]

        for r in range(1, self.cumulative_energy_image.shape[0]):
            for c in range(0, self.cumulative_energy_image.shape[1]):
                lowest = 99999
                for i in range(-1, 2):
                    if 0 <= c + i < self.cumulative_energy_image.shape[1]:
                        if self.cumulative_energy_image[r - 1, c + i] < lowest:
                            lowest = self.cumulative_energy_image[r - 1, c + i]
                self.cumulative_energy_image[r, c] = lowest + energy_image[r, c]

        return self.cumulative_energy_image

    def find_seam_locations(self) -> List[int]:
        """
        Given a filled-in cumulative grid, finds the vertical seam corresponding to the least energy used.
        :param energy_image: the energy image for the source image.
        :param cumulative_energy_image: a filled-in cumulative grid
        :return: a list of the column numbers for seam location of each row. So seam_values = [ 12, 13, 13, 14, 15, 14]
        would correspond to a seam consisting of (r, c) points (0, 12), (1, 13), (2, 13), (3, 14), (4, 15), (5, 14).
        """

        # Finds the index of the lowest item in the bottom row of the graphic.
        # I THINK YOU'LL FIND THIS HANDY.
        minstart_x: int = int(np.argmin(self.cumulative_energy_image[-1, :]))
        height, width = self.cumulative_energy_image.shape

        seam_values = [0] * height
        seam_values[0] = minstart_x

        r = height - 2
        while r >= 0:
            seam_index = height - r - 1
            lowest = 99999
            for i in range(-1, 2):
                if 0 <= seam_values[seam_index - 1] + i < width:
                    if self.cumulative_energy_image[r, seam_values[seam_index - 1] + i] < lowest:
                        lowest = self.cumulative_energy_image[r, seam_values[seam_index - 1] + i]
                        seam_values[seam_index] = seam_values[seam_index - 1] + i
            r -= 1

        seam_values.reverse()
        return seam_values

    def build_seam_image_with_path(self, source_image: np.ndarray,
                                seam_values: List[int]) -> np.ndarray:
        """
        given a list of the column numbers for a path from the top row to the bottom row, creates an image with a bw
        copy of the source and a red line representing the seam.
        :param source_image: the color image we will draw this seam onto.
        :param seam_values: a list of integers, corresponding to the horizontal (col) location for each point on a line
        extending from the top of the image to the bottom. (The length of this list should be the same as the height
        of the image.)
        :return: new seam image.
        """
        if len(seam_values) != source_image.shape[0]:
            raise RuntimeError(f"Error - seam list is different height than image. " \
                                 f"{len(seam_values)=}\t{source_image.shape[0]=}")
        seam_image = source_image.copy()

        num_r = seam_image.shape[0]
        for r in range(0, num_r):
            seam_image[r, seam_values[r]] = (0, 0, 255)

        return seam_image

    def remove_seam_from_image(self, seam_values: List[int],
                                     source_image: np.ndarray) -> np.ndarray:
        """
        remove the pixels corresponding the horizontal positions in the seam_values list from the source_image. Any
        pixels to the left of the value stored in seam_values for a given row will stay the same, but any pixels to
        the right of the seam_value for this row will be shifted one pixel to the left.
        :param seam_values: a list of integers, one per row, indicating which pixel in the row to remove.
        :param source_image: the image from which to remove (this image is not altered.)
        :return: the resulting image, which will be one pixel narrower than the source image.
        """
        if len(seam_values) != source_image.shape[0]:
            raise RuntimeError(f"Error - seam list is different height than image. " \
                                    f"{len(seam_values)=}\t{source_image.shape[0]=}")

        result_image = source_image.copy()
        # result_energy = self.energy_image.copy()
        result_cumulative = self.cumulative_energy_image.copy()
        for r in range(source_image.shape[0]):
            # copy the second "half" of the row to a range one pixel to the left.
            result_image[r, seam_values[r]:-1] = result_image[r, seam_values[r] + 1:]
            # result_energy[r, seam_values[r]:-1] = result_energy[r, seam_values[r] + 1:]
            result_cumulative[r, seam_values[r]:-1] = result_cumulative[r, seam_values[r] + 1:]

        result_image = result_image[:, :-1] # crop the last column.
        # self.energy_image = result_energy[:, :-1]
        self.cumulative_energy_image = result_cumulative[:, :-1]
        return result_image

    # @log_start_stop_method
    def recalculate_altered_cumulative_energy_grid(self, seam_values):
        # initial values are redundant as will be reassigned shortly
        spike_start = seam_values[0]
        spike_end = seam_values[0] + 1

        for r in range(0, self.energy_image.shape[0]):
            spike_start = min(spike_start, seam_values[r])
            spike_end = max(spike_end, min(seam_values[r] + 1, self.energy_image.shape[1]))

            if spike_start > 0 and self.energy_image[r, spike_start - 1] > self.energy_image[r, spike_start]:
                spike_start -= 1
            if spike_end < self.energy_image.shape[1] and self.energy_image[r, spike_end] > self.energy_image[r, spike_end - 1]:
                spike_end += 1

            for c in range(spike_start, spike_end):
                lowest = 99999
                for i in range(-1, 2):
                    if 0 <= c + i < self.cumulative_energy_image.shape[1]:
                        if self.cumulative_energy_image[r - 1, c + i] < lowest:
                            lowest = self.cumulative_energy_image[r - 1, c + i]
                self.cumulative_energy_image[r, c] = lowest + self.energy_image[r, c]




