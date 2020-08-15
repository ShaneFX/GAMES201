import numpy as np
from plyfile import PlyData


class PlyImporter:
    def __init__(self, file):
        self.file_name = file
        plydata = PlyData.read(self.file_name)
        data = plydata['vertex'].data
        self.count = plydata['vertex'].count
        self.np_array = np.array([[x, y, z] for x, y, z in data])

    def get_array(self):
        return self.np_array

    def get_count(self):
        return self.count

    def multiply(self, mul):
        self.np_array *= mul


if __name__ == "__main__":
    ply1 = PlyImporter("test.ply")
    ply1.multiply(0.5)
    np1 = ply1.get_array()
    np2 = np1.reshape((4, 4, 4, 3))
    print(np2)
