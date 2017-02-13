import numpy as np

class MDEngine:

    def __init__(self, temp=120, position_file=None):
        self.temp = temp
        if position_file is None:
            # generate positions

    # face centered cubic
    def generate_positions(self, num_cells=8, length=1.0):
        num_particles = 4*num_cells**3
        positions_mat = np.zeros((num_particles, 3))

    def position_setup(ifn):
        ifile = open(ifn, 'r')

