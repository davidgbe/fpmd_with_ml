import csv
from lib.gaussian_process import utilities
import numpy as np
from lib.internal_vector import utilities as iv_utilities
from lib.gaussian_process.model import GaussianProcess as GP
from numpy.linalg import pinv
import sys

class MDForcesPredictor:
    @staticmethod
    def predict(data_path):
        start = 1000
        end = 1200
        internal_reps = MDForcesPredictor.load_data(data_path + '/iv_reps_108.txt', start, end)
        forces = MDForcesPredictor.load_data(data_path + '/forcefile_7000step_108part.txt', start, end)
        forces_k_space = MDForcesPredictor.convert_forces_to_internal(forces, internal_reps)
        feature_mats = MDForcesPredictor.produce_feature_mats(internal_reps)

        gp = GP()

        print(feature_mats.shape)
        print(forces_k_space.shape)

        training_test_divide = 180

        predictions = gp.predict(feature_mats[:training_test_divide], forces_k_space[:training_test_divide], feature_mats[training_test_divide:])
        predicted_cart_forces = MDForcesPredictor.convert_internal_forces_to_cartesian(predictions, internal_reps)
        print(len(forces[training_test_divide:]))
        print(len(predicted_cart_forces))
        for real_example, predicted in zip(forces[training_test_divide:], predicted_cart_forces):
            for real_forces, predicted_forces in zip(real_example, predicted): 
                print('Example:')
                print(real_forces)
                print(predicted_forces)

    @staticmethod
    def produce_internal():
        start = 0
        end = 100
        internal_reps = MDForcesPredictor.load_arrangements_in_internal('../datasets/md/posfile_7000step_108part.txt', start, end)

        MDForcesPredictor.write_data('../datasets/md/iv_reps_108.txt', internal_reps)

    @staticmethod
    def produce_feature_mats(internal_reps):
        return np.concatenate(list(map(lambda x: iv_utilities.produce_feature_matrix(x), internal_reps)), axis=0)

    @staticmethod
    def convert_forces_to_internal(forces, internal_reps):
        forces_k_space = []
        num_forces_per_arrangement = forces[0].shape[0]
        for i in range(len(forces)):
            forces_for_arrangement = []
            for j in range(num_forces_per_arrangement):
                forces_for_arrangement.append(internal_reps[i].dot(forces[i][j].T))
            forces_for_arrangement = [ force_component for forces in forces_for_arrangement for force_component in forces ]
            forces_k_space.append(np.array(forces_for_arrangement).reshape(1, len(forces_for_arrangement)))
        return np.concatenate(forces_k_space, axis=0)

    @staticmethod
    def convert_internal_forces_to_cartesian(forces, internal_reps):
        cart_forces = []
        k = int(np.sqrt(forces.shape[1]))
        for i in range(len(forces)):
            internal_inv = pinv(internal_reps[i])
            cart_forces.append([ internal_inv.dot(forces[i][j:j+k]) for j in range(0, forces.shape[1], k) ])
        return cart_forces

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

MDForcesPredictor.predict(sys.argv[1])
