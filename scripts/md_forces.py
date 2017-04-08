import csv
from lib.gaussian_process import utilities
import numpy as np
from lib.internal_vector import utilities as iv_utilities

class MDForcesPredictor:
    @staticmethod
    def run():
        positions = MDForcesPredictor.load_data('../datasets/md/posfile_5000step_256part.txt', 10, 100)
        forces = MDForcesPredictor.load_data('../datasets/md/forcefile_5000step_256part.txt', 10, 100)

        for arrangement in positions:
            print(iv_utilities.produce_internal_basis(arrangement))

    @staticmethod
    def load_data(rel_path, start=0, end=None):
        data_file = open(utilities.file_path(__file__, rel_path), 'r')
        parsed_csv = csv.reader(data_file, delimiter=',')
        all_data = []

        arrangement_data = []
        for row in parsed_csv:
            if row[0] != 'NEW':
                arrangement_data.append(row)
            if row[0] == 'NEW':
                if len(arrangement_data) != 0:
                    all_data.append(np.array(arrangement_data))
                    if end is not None and len(all_data) == end:
                        return all_data[start:]
                    arrangement_data = []
        return all_data[start:]

MDForcesPredictor.run()
