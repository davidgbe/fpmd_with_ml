import csv
from lib.gaussian_process import utilities
import numpy as np
from lib.internal_vector import utilities as iv_utilities
from lib.gaussian_process.model import GaussianProcess as GP

class MDForcesPredictor:
    @staticmethod
    def predict():
        start = 0
        end = 99
        internal_reps = MDForcesPredictor.load_data('../datasets/md/iv_reps_2.txt')
        forces = MDForcesPredictor.load_data('../datasets/md/forcefile_5000step_256part.txt', start, end)
        forces_k_space = MDForcesPredictor.convert_forces_to_internal(forces, internal_reps)
        feature_mats = MDForcesPredictor.produce_feature_mats(internal_reps)

        gp = GP()

        print(feature_mats.shape)
        print(forces_k_space.shape)

        predictions = gp.predict(feature_mats[:90], forces_k_space[:90], feature_mats[98:])
        print(predictions)

    @staticmethod
    def produce_internal():
        start = 0
        end = 100
        internal_reps = MDForcesPredictor.load_arrangements_in_internal('../datasets/md/posfile_5000step_256part.txt', start, end)

        MDForcesPredictor.write_data('../datasets/md/iv_reps_2.txt', internal_reps)

    @staticmethod
    def produce_feature_mats(internal_reps):
        return np.concatenate(list(map(lambda x: iv_utilities.produce_feature_matrix(x), internal_reps)), axis=0)

    @staticmethod
    def convert_forces_to_internal(forces, internal_reps):
        forces_k_space = []
        for i in range(len(forces)):
            forces_k_space.append(internal_reps[i].dot(forces[i].T).reshape(1, (internal_reps[i].shape[0])**2))
        return np.concatenate(forces_k_space, axis=0)

    @staticmethod
    def load_arrangements_in_internal(rel_path, start=0, end=None):
        positions = MDForcesPredictor.load_data(rel_path, start, end)
        count = 1
        internal_reps = []
        for arrangement in positions:
            internal_reps.append(iv_utilities.produce_internal_basis(arrangement))
            print(count)
            count += 1
        return internal_reps

    @staticmethod
    def load_data(rel_path, start=0, end=None):
        data_file = open(utilities.file_path(__file__, rel_path), 'r')
        parsed_csv = csv.reader(data_file, delimiter=',')
        all_data = []

        arrangement_data = []
        for row in parsed_csv:
            if row[0] != 'NEW':
                arrangement_data.append(list(map(lambda x: float(x), row)))
            if row[0] == 'NEW':
                if len(arrangement_data) != 0:
                    all_data.append(np.array(arrangement_data))
                    if end is not None and len(all_data) == end:
                        return all_data[start:]
                    arrangement_data = []
        data_file.close()
        return all_data[start:]

    @staticmethod
    def write_data(rel_path, data):
        data_file = open(utilities.file_path(__file__, rel_path), 'w')
        csv_writer = csv.writer(data_file, delimiter=',')
        for example in data:
            print(example)
            data_file.write('NEW\n')
            for i in range(example.shape[0]):
                csv_writer.writerow(example[i])
        data_file.close()

MDForcesPredictor.predict()
