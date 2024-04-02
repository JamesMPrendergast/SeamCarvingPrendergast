import unittest
import numpy as np

from CarverFile import Carver

class MyTestCase(unittest.TestCase):
    # def setUp(self) -> None:


    def test_a_small_cumulative_energy_grid(self):
        nrg = np.array([[37, 67, 66, 48],
                         [73, 21, 70, 51],
                         [32, 48, 65, 65],
                         [60, 57, 48, 23],
                         [55, 41, 70, 79],
                         [42, 78, 52, 23]])

        expected_cumulative = np.array([[ 37, 67, 66, 48],
                         [110, 58,118, 99],
                         [ 90,106,123,164],
                         [150,147,154,146],
                         [202,188,216,225],
                         [230,266,240,239]])

        carver = Carver()

        result = carver.generate_cumulative_energy_grid(energy_image=nrg).astype(int)
        print("Test A - cumulative energy grid:")
        print(result)

        self.assertSequenceEqual(expected_cumulative.tolist(), result.tolist())

    def test_b_large_cumulative_energy_grid(self):
        nrg = np.array([[73, 51, 79, 46, 23, 32, 77, 39],
                       [22, 66, 80, 48, 54, 72, 46, 40],
                       [38, 42, 26, 29, 51, 78, 60, 80],
                       [61, 64, 61, 20, 60, 23, 41, 79],
                       [55, 52, 66, 51, 66, 76, 44, 67],
                       [50, 32, 57, 34, 44, 69, 74, 22],
                       [31, 72, 64, 44, 22, 47, 31, 52],
                       [26, 60, 62, 79, 73, 28, 72, 26]])


        expected_cumulative = np.array([[73, 51, 79, 46, 23, 32, 77, 39],
        [73, 117, 126, 71, 77, 95, 78, 79],
        [111, 115, 97, 100, 122, 155, 138, 158],
        [172, 161, 158, 117, 160, 145, 179, 217],
        [216, 210, 183, 168, 183, 221, 189, 246],
        [260, 215, 225, 202, 212, 252, 263, 211],
        [246, 287, 266, 246, 224, 259, 242, 263],
        [272, 306, 308, 303, 297, 252, 314, 268]])

        carver = Carver()

        result = carver.generate_cumulative_energy_grid(energy_image=nrg).astype(int)
        print("Test B - cumulative energy grid:")
        print (result)

        self.assertSequenceEqual(expected_cumulative.tolist(), result.tolist())

    def test_c_short_path(self):
        nrg = np.array([[37, 67, 66, 48],
                        [73, 21, 70, 51],
                        [32, 48, 65, 65],
                        [60, 57, 48, 23],
                        [55, 41, 70, 79],
                        [42, 78, 52, 23]])
        cumulative_nrg = np.array([[37, 67, 66, 48],
                                        [110, 58, 118, 99],
                                        [90, 106, 123, 164],
                                        [150, 147, 154, 146],
                                        [202, 188, 216, 225],
                                        [230, 266, 240, 239]])

        expected_path = [0, 1, 0, 1, 1, 0]
        carver = Carver()
        result = carver.find_seam_locations(nrg, cumulative_nrg)
        print(f"Test C: path found: {result}")

        self.assertListEqual(expected_path, result)

    def test_d_long_path(self):
        nrg = np.array([[73, 51, 79, 46, 23, 32, 77, 39],
                        [22, 66, 80, 48, 54, 72, 46, 40],
                        [38, 42, 26, 29, 51, 78, 60, 80],
                        [61, 64, 61, 20, 60, 23, 41, 79],
                        [55, 52, 66, 51, 66, 76, 44, 67],
                        [50, 32, 57, 34, 44, 69, 74, 22],
                        [31, 72, 64, 44, 22, 47, 31, 52],
                        [26, 60, 62, 79, 73, 28, 72, 26]])

        cumulative_nrg = np.array([[73, 51, 79, 46, 23, 32, 77, 39],
                                        [73, 117, 126, 71, 77, 95, 78, 79],
                                        [111, 115, 97, 100, 122, 155, 138, 158],
                                        [172, 161, 158, 117, 160, 145, 179, 217],
                                        [216, 210, 183, 168, 183, 221, 189, 246],
                                        [260, 215, 225, 202, 212, 252, 263, 211],
                                        [246, 287, 266, 246, 224, 259, 242, 263],
                                        [272, 306, 308, 303, 297, 252, 314, 268]])
        expected_path = [4, 3, 2, 3, 3, 3, 4, 5]
        carver = Carver()
        result = carver.find_seam_locations(nrg, cumulative_nrg)
        print(f"Test D: path found: {result}")
        self.assertListEqual(expected_path, result)

    def test_e_find_path_from_large_energy(self):
        nrg = np.array([[23, 78, 45, 21, 35, 62, 23, 60, 39, 31, 38, 70, 33, 39, 32],
                                         [67, 21, 56, 78, 70, 64, 64, 49, 23, 59, 38, 77, 59, 37, 51],
                                         [51, 72, 56, 54, 74, 52, 43, 36, 78, 56, 65, 30, 51, 28, 22],
                                         [60, 62, 21, 71, 79, 33, 54, 50, 67, 64, 79, 22, 69, 30, 53],
                                         [27, 79, 50, 46, 21, 38, 78, 34, 72, 39, 54, 33, 61, 56, 44],
                                         [23, 73, 76, 49, 72, 61, 52, 71, 71, 28, 66, 62, 62, 51, 57],
                                         [29, 74, 47, 70, 67, 80, 26, 54, 79, 65, 41, 35, 76, 80, 36],
                                         [31, 58, 30, 68, 72, 50, 28, 67, 23, 36, 41, 31, 30, 70, 58],
                                         [43, 50, 66, 55, 28, 61, 32, 24, 33, 22, 58, 31, 72, 63, 61],
                                         [30, 40, 68, 70, 54, 38, 23, 21, 59, 21, 72, 20, 56, 21, 42],
                                         [43, 71, 26, 39, 49, 64, 64, 57, 27, 79, 20, 68, 63, 68, 39],
                                         [40, 73, 51, 33, 40, 21, 76, 45, 26, 40, 78, 43, 43, 30, 58],
                                         [34, 50, 49, 57, 75, 78, 51, 51, 33, 57, 74, 78, 48, 27, 38],
                                         [79, 20, 56, 35, 30, 50, 63, 75, 56, 62, 74, 29, 44, 75, 34],
                                         [50, 35, 72, 33, 52, 33, 43, 32, 54, 63, 43, 43, 72, 60, 71]])
        expected_path = [9, 10, 11, 11, 10, 9, 10, 9, 9, 9, 8, 8, 8, 8, 7]
        carver = Carver()
        cumulative_nrg = carver.generate_cumulative_energy_grid(energy_image=nrg).astype(int)
        result_path = carver.find_seam_locations(nrg, cumulative_nrg)

        print(f"Test E: path found: {result_path}")
        self.assertListEqual(expected_path, result_path)

if __name__ == '__main__':
    unittest.main()
