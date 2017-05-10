import csv
from lib.gaussian_process import utilities
import numpy as np
from lib.internal_vector import utilities as iv_utilities
from lib.gaussian_process.model import GaussianProcess as GP
from numpy.linalg import pinv
import sys
import gc
import matplotlib.pyplot as plt

class MDForcesPredictor:

    def predict(arrangements):
        pass

    def fit(arrangements, forces):
        internal_reps = MDForcesPredictor.produce_internal_from_arrangements(arrangements)
        internal_reps_normed = [ iv_utilities.normalize_mat(rep) for rep in internal_reps ]
        forces_k_space = MDForcesPredictor.convert_forces_to_internal(forces, internal_reps_normed)
        feature_mats = MDForcesPredictor.produce_feature_mats(internal_reps)

        num_examples = len(internal_reps)

        del internal_reps

        training_examples = [feature_mats, internal_reps_normed, forces_k_space]
        (feature_mats_training, internal_reps_normed_training, forces_k_space_training) = utilities.sample_populations(training_examples, size=num_examples)[0]

        # get rid of unused examples
        del training_examples
        del feature_mats
        del internal_reps_normed
        del forces
        del forces_k_space

        self.gp = GP()

    @staticmethod
    def predict(data_path, training_size=1000):
        start = 2600
        end = 6400

        # write first number of first arrangement used to make internal rep data in file
        internal_reps = MDForcesPredictor.load_data(data_path + '/iv_reps_108_1_to_6_half.txt', start - 1000, end - 1000)
        internal_reps_normed = [ iv_utilities.normalize_mat(rep) for rep in internal_reps ]
        forces = MDForcesPredictor.load_data(data_path + '/for_108_7000TEST.txt', start, end)
        forces_k_space = MDForcesPredictor.convert_forces_to_internal(forces, internal_reps_normed)
        feature_mats = MDForcesPredictor.produce_feature_mats(internal_reps)

        del internal_reps

        # split into training and testing populations
        to_sample = [feature_mats, internal_reps_normed, forces, forces_k_space]
        num_to_test = 200
        (testing, training) = utilities.sample_populations(to_sample, size=num_to_test, remove=True)
        (feature_mats_testing, internal_reps_normed_testing, forces_testing, forces_k_space_testing) = testing
        (feature_mats_training, internal_reps_normed_training, forces_training, forces_k_space_training) = utilities.sample_populations(training, size=training_size)[0]

        utilities.print_memory()

        # get rid of unused examples
        del training
        del feature_mats
        del internal_reps_normed
        del forces
        del forces_k_space
        gc.collect()

        utilities.print_memory()

        gp = GP()

        predictions = gp.predict(feature_mats_training, forces_k_space_training, feature_mats_testing)
        predicted_cart_forces = MDForcesPredictor.convert_internal_forces_to_cartesian(predictions, internal_reps_normed_testing)

        errors = []

        predicted_cart_forces = np.array([ force_vec for forces_for_arrangement in predicted_cart_forces for force_vec in forces_for_arrangement ])
        forces_testing = np.array([ force_vec for forces_for_arrangement in forces_testing for force_vec in forces_for_arrangement ])

        MDForcesPredictor.calc_force_error(forces_testing, predicted_cart_forces)

    @staticmethod
    def produce_internal():
        start = 1000
        end = 6500
        internal_reps = MDForcesPredictor.load_arrangements_in_internal('../datasets/md/posfile_7000step_108part.txt', start, end)
        MDForcesPredictor.write_data('../datasets/md/iv_reps_108_all.txt', internal_reps)

    @staticmethod
    def load_new_data():
        all_positions, all_forces = MDForcesPredictor.alternate_load_data('../datasets/lj.dat', 0, 500)

        first_step = all_forces[0]

        first_step_shortened = first_step[:108, :]
        print(first_step_shortened)

        last_step = all_forces[499]

        last_step_shortened = first_step[:108, :]

        all_steps = np.concatenate(all_forces)

        all_steps_shortened = []
        for i in range(0, len(all_forces), 1000):
            all_steps_shortened.append(all_steps[i:i+108])
        all_steps_shortened = np.concatenate(all_steps_shortened)


        MDForcesPredictor.plot_force_dist(first_step, 'first')
        MDForcesPredictor.plot_force_dist(last_step, 'last')
        MDForcesPredictor.plot_force_dist(first_step_shortened, 'first_shortened')
        MDForcesPredictor.plot_force_dist(last_step_shortened, 'last_shortened')

        MDForcesPredictor.plot_force_dist(all_steps, 'all')
        MDForcesPredictor.plot_force_dist(all_steps_shortened, 'all_shortened')



    @staticmethod
    def plot_force_dist(step, name):
        for i in range(step.shape[1]):
            coordinate_mags = step[:, i]
            (mags, freq) = utilities.bucket(coordinate_mags, .05)
            plt.plot(mags, freq, 'ro')
            utilities.save_plot('dist_' + name + '_' + str(i))
            plt.clf()

    @staticmethod
    def produce_feature_mats(internal_reps):
        return np.concatenate(list(map(lambda x: iv_utilities.produce_feature_matrix(x), internal_reps)), axis=0)

    @staticmethod
    def convert_forces_to_internal(forces, internal_reps_normed):
        forces_k_space = []
        num_forces_per_arrangement = forces[0].shape[0]
        for i in range(len(forces)):
            forces_for_arrangement = []
            for j in range(num_forces_per_arrangement):
                forces_for_arrangement.append(internal_reps_normed[i].dot(forces[i][j].T))
            forces_for_arrangement = [ force_component for forces in forces_for_arrangement for force_component in forces ]
            forces_k_space.append(np.array(forces_for_arrangement).reshape(1, len(forces_for_arrangement)))
        return np.concatenate(forces_k_space, axis=0)

    @staticmethod
    def convert_internal_forces_to_cartesian(forces, internal_reps_normed):
        cart_forces = []
        k = int(np.sqrt(forces.shape[1]))
        for i in range(len(forces)):
            internal_inv = pinv(internal_reps_normed[i])
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
    def produce_internal_from_arrangements(positions):
        internal_reps = []
        for arrangement in positions:
            internal_reps.append(iv_utilities.produce_internal_basis(arrangement))
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

    # for Guoqing's file format
    @staticmethod
    def alternate_load_data(rel_path, start=0, end=None):
        data_file = open(utilities.file_path(__file__, rel_path), 'r')
        parsed_csv = csv.reader(data_file, delimiter=' ')

        all_positions = []
        all_forces = []

        arrangement_positions = []
        arrangement_forces = []
        reading = False
        count = 0
        for row in parsed_csv:
            if ' '.join(row).startswith('ITEM: ATOMS id x y z fx fy fz'):
                reading = True
                count += 1
            elif reading == True:
                if row[0] == 'ITEM:':
                    reading = False
                    all_positions.append(np.array(arrangement_positions))
                    all_forces.append(np.array(arrangement_forces))
                    arrangement_positions = []
                    arrangement_forces = []
                    if count == end:
                        return all_positions, all_forces
                else:
                    arrangement_positions.append(list(map(lambda x: float(x), row[1:4])))
                    arrangement_forces.append(list(map(lambda x: float(x), row[4:7])))
        return all_positions, all_forces

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

    @staticmethod
    def calc_force_error(actual_forces, predicted, error_tolerance=0.03):
        #err_file = open("../err_" + error_tolerance + "_" + actual_forces.size + ".txt", 'w')
        errors = []
        thresholds = 2 * np.absolute(actual_forces).mean(0) * error_tolerance
        for real_force_vec, predicted_force_vec in zip(actual_forces, predicted):
            print('vector:')
            for i in range(3):

### DAVID: there are two options for the file writing, the first, which is not commented out, creates
### an error file tracking all forces and errors, but only printing those that are not cut out by the
### threshold you set. I think this is preferable as it allows us to retain all of the data, but if
### you'd rather only keep track of what we're printing, there is a commented out second version below

                error = abs(predicted_force_vec[i] - real_force_vec[i]) / abs(real_force_vec[i]) * 100
                # Added line below for writing values and error to error file
                # err_file.write(str(real_force_vec[i]) + "," + str(predicted_force_vec[i]) + "," + str(error) + "\n")
                if abs(real_force_vec[i]) > thresholds[i]:
                    print(real_force_vec[i])
                    print(predicted_force_vec[i])
                    errors.append(error)

                # if abs(real_force_vec[i]) > thresholds[i]:
                #     print(real_force_vec[i])
                #     print(predicted_force_vec[i])
                #     error = abs(predicted_force_vec[i] - real_force_vec[i]) / abs(real_force_vec[i]) * 100
                #     errors.append(error)
                #     # Added line below for writing values and error to error file
                #     err_file.write(str(real_force_vec[i]) + "," + str(predicted_force_vec[i]) + "," + str(error) + "\n")

        print(errors)
        print(np.median(errors))
        print(np.average(errors))

#MDForcesPredictor.predict(sys.argv[1], int(sys.argv[2]))
#MDForcesPredictor.produce_internal()
MDForcesPredictor.load_new_data()
