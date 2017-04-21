import sys
import numpy as np
from numpy.linalg import norm as mag
from numpy.random import random

## GENERATORS ##

# Generator for 3 for loops of same size
def triple_for(size):
    for i in range(size):
        for j in range(size):
            for k in range(size):
                yield (i,j,k)

# Generator for setting x, y, z in a matrix
def set_matrix_x_y_z(matrix, index, x_val, y_val, z_val):
    matrix[index,0] = x_val
    matrix[index,1] = y_val
    matrix[index,2] = z_val

# Generator for adding to x, y, z in a matrix
def add_matrix_x_y_z(matrix, index, x_val, y_val, z_val):
    matrix[index,0] += x_val
    matrix[index,1] += y_val
    matrix[index,2] += z_val


## Engine specific ##

# To get a normal (Gaussian) distibution of random numbers
# Based on "basic form" listed here: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
def rand_box_muller():
    u1 = random()
    u2 = random()
    r = np.sqrt((-2.)*np.log(u1))
    t = 2.*np.pi*u2
    return r*np.cos(t)

# To get the magnitude of the distance between two particles
def particles_distance(positions_mat, particle1, particle2, L_star):
    x_diff = np.abs(positions_mat[particle1,0]-positions_mat[particle2,0])
    x_diff = np.where(x_diff < 0.5*L_star, x_diff, L_star-x_diff)
    
    y_diff = np.abs(positions_mat[particle1,1]-positions_mat[particle2,1])
    y_diff = np.where(y_diff < 0.5*L_star, y_diff, L_star-y_diff)
    
    z_diff = np.abs(positions_mat[particle1,2]-positions_mat[particle2,2])
    z_diff = np.where(z_diff < 0.5*L_star, z_diff, L_star-z_diff)
    
    return np.sqrt(x_diff**2. + y_diff**2. + z_diff**2.)

# To get x, y, or z component of distance between two particles, over the magnitude of their total distance (signed)
def particle_component_ratio(positions_mat, component, particle1, particle2, L_star):
    total_distance = particles_distance(positions_mat, particle1, particle2, L_star)
    component_distance = positions_mat[particle1,component]-positions_mat[particle2,component]
    
    if(component_distance > 0.5*L_star):
        component_distance -= L_star
    elif (component_distance < -0.5*L_star):
        component_distance += L_star
    if(total_distance == 0):
        return 0
    
    return component_distance/total_distance

# To get the total kinetic energy
def find_kin_energy(tau_velo_mat):
    return 0.5*np.sum([ mag(tau_velo_mat[i])**2. for i in range(tau_velo_mat.shape[0]) ])

# To find a particle's correct component position in-case boundary conditions require it
def position_fix(component_position, L_star):
    if(component_position > L_star):
        component_position -= L_star
    elif(component_position < 0):
        component_position += L_star
    return component_position

# To get the distance between two particles in a single component, considering boundary conditions
def boundary_cond_dist(position1, position2, L_star):
    distance = position1-position2
    if(distance > 0.5*L_star):
        distance -= L_star
    elif (distance < -0.5*L_star):
        distance += L_star
    return distance

# The engine!
class MDEngine:
    # Initialization
    def __init__(self, num_particles = 256, out_file_name = None):

        self.num_particles = num_particles                                  # Number of particles
        self.T_star = 1.                                                    # Dimensionless Temperature
        self.rho_star = 0.8                                                 # Dimensionless particle number density
        self.L_star = (self.num_particles/self.rho_star)**(1./3)            # Dimensionless length of lattice
        self.r_c_star = 2.5                                                 # Dimensionless cut off length
        self.del_t = 0.005                                                  # Dimensionless time

        # Open files for position and force printing
        self.posfile = open("../engine_outputs/pos_quickTEST.txt", 'w')
        self.forcefile = open("../engine_outputs/for_quickTEST.txt", 'w')

        # Open file for energies to be written to
        self.has_ofile = 0
        if(out_file_name != None):
            self.has_ofile = 1
            self.ofile = open(out_file_name, 'w')

        # Generate matrices to get started
        self.generate_positions()
        self.generate_tau_velos()

        # Calculate and print initial values
        self.calculate_1st_total_energies()
        if(self.has_ofile):
            self.ofile.write(str(self.total_kin_energy/self.num_particles))
            self.ofile.write(",")
            self.ofile.write(str(self.total_pot_energy/self.num_particles))
            self.ofile.write(",")
            self.ofile.write(str(self.energy_per_particle))
            self.ofile.write(",")
            self.ofile.write(str((2./3)*find_kin_energy(self.tau_velo_mat)/(self.num_particles)))
            self.ofile.write("\n")


        # Do first step, which is done differently due to not having position history
        self.calculate_tau_accels_pot_energy()
        self.calculate_1st_new_pos_velos()
        # Fix the tau_velos for current temperature
        self.fix_tau_velos()
        self.calculate_total_energies()
        if(self.has_ofile):
            self.ofile.write(str(self.total_kin_energy/self.num_particles))
            self.ofile.write(",")
            self.ofile.write(str(self.total_pot_energy/self.num_particles))
            self.ofile.write(",")
            self.ofile.write(str(self.energy_per_particle))
            self.ofile.write(",")
            self.ofile.write(str((2./3)*find_kin_energy(self.tau_velo_mat)/(self.num_particles)))
            self.ofile.write("\n")


    # Generate face centered cubic positions for particles
    def generate_positions(self):
        num_cells_per_edge = int(round((self.num_particles/4.)**(1./3.)))
        cell_length = self.L_star/(num_cells_per_edge)
        positions_mat = np.zeros((self.num_particles, 3))
        #orig_pos_mat = np.zeros((self.num_particles, 3))
        prev_pos_mat = np.zeros((self.num_particles, 3))
        i = 0;

        # add all cell-corner particles (should be first num_cells^3 particles)
        for j, k, l in triple_for(num_cells_per_edge):
            set_matrix_x_y_z(positions_mat, i, j*cell_length, k*cell_length, l*cell_length)
            #set_matrix_x_y_z(orig_pos_mat, i, j*cell_length, k*cell_length, l*cell_length)
            set_matrix_x_y_z(prev_pos_mat, i, j*cell_length, k*cell_length, l*cell_length)
            i += 1

        # add all face particles with n*cell_length x values
        for j, k, l in triple_for(num_cells_per_edge):
            set_matrix_x_y_z(positions_mat, i, j*cell_length, (k+0.5)*cell_length, (l+0.5)*cell_length)
            #set_matrix_x_y_z(orig_pos_mat, i, j*cell_length, (k+0.5)*cell_length, (l+0.5)*cell_length)
            set_matrix_x_y_z(prev_pos_mat, i, j*cell_length, (k+0.5)*cell_length, (l+0.5)*cell_length)
            i += 1

        # add all face particles with n*cell_length y values
        for j, k, l in triple_for(num_cells_per_edge):
            set_matrix_x_y_z(positions_mat, i, (j+0.5)*cell_length, k*cell_length, (l+0.5)*cell_length)
            #set_matrix_x_y_z(orig_pos_mat, i, (j+0.5)*cell_length, k*cell_length, (l+0.5)*cell_length)
            set_matrix_x_y_z(prev_pos_mat, i, (j+0.5)*cell_length, k*cell_length, (l+0.5)*cell_length)
            i += 1

        # add all face particles with n*cell_length z values
        for j, k, l in triple_for(num_cells_per_edge):
            set_matrix_x_y_z(positions_mat, i, (j+0.5)*cell_length, (k+0.5)*cell_length, l*cell_length)
            #set_matrix_x_y_z(orig_pos_mat, i, (j+0.5)*cell_length, (k+0.5)*cell_length, l*cell_length)
            set_matrix_x_y_z(prev_pos_mat, i, (j+0.5)*cell_length, (k+0.5)*cell_length, l*cell_length)
            i += 1

        self.positions_mat = positions_mat
        #self.orig_pos_mat = orig_pos_mat
        self.prev_pos_mat = prev_pos_mat

    # Fix the tau_velos so they still correspond to the T_star temperature
    def fix_tau_velos(self):
        # Find the temperature that these tau_velos correspond to, and the needed fix
        T_star_prime = (2./3)*find_kin_energy(self.tau_velo_mat)/(self.num_particles)
        tau_velo_fix = np.sqrt(self.T_star/T_star_prime)

        # Fix the tau_velos so they now lead to a temperature of T_star
        for i in range(self.num_particles):
            for j in range(3):
                self.tau_velo_mat[i,j] = self.tau_velo_mat[i,j]*tau_velo_fix

    # Generate random "tau_velo" (tau times velocity vector) for each particle, and fix such that total temperature is correct
    def generate_tau_velos(self):
        # Set up matrix of randomized tau_velos in x, y, and z direction
        self.tau_velo_mat = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            for j in range(3):
                self.tau_velo_mat[i,j] = rand_box_muller()

        # Fix the tau_velos for current temperature
        self.fix_tau_velos()

    # Calculate the "tau_accel" on each particle, that is tau^2 times the acceleration vector, and total potential energy
    def calculate_tau_accels_pot_energy(self):
        # Matrix of components of tau^2 times acceleration
        tau_accel_mat = np.zeros((self.num_particles, 3))
        total_pot_energy = 0

        # For each particle, find the total "tau_accel" of all other particles on it
        for i in range(self.num_particles):
            for j in range(i, self.num_particles):
                if(i != j):
                    r_ij = particles_distance(self.positions_mat,i,j,self.L_star)
                    r_ji = particles_distance(self.positions_mat,j,i,self.L_star)
                    if(r_ij > self.r_c_star):
                        pass
                    else:
                        total_pot_energy += 0.5 * ((1./r_ij)**12. - (1./r_ij)**6.)
                        total_pot_energy += 0.5 * ((1./r_ji)**12. - (1./r_ji)**6.)

                        tau_accel_ij_magnitude = 12. * (1./r_ij**6.) * ((1./r_ij**6.) - 0.5)
                        tau_accel_ji_magnitude = 12. * (1./r_ji**6.) * ((1./r_ji**6.) - 0.5)
                        # Find signed component "tau_accels"
                        x_tau_accel_ij = tau_accel_ij_magnitude*particle_component_ratio(self.positions_mat,0,i,j,self.L_star)
                        y_tau_accel_ij = tau_accel_ij_magnitude*particle_component_ratio(self.positions_mat,1,i,j,self.L_star)
                        z_tau_accel_ij = tau_accel_ij_magnitude*particle_component_ratio(self.positions_mat,2,i,j,self.L_star)
                        add_matrix_x_y_z(tau_accel_mat, i, x_tau_accel_ij, y_tau_accel_ij, z_tau_accel_ij)

                        x_tau_accel_ji = tau_accel_ji_magnitude*particle_component_ratio(self.positions_mat,0,j,i,self.L_star)
                        y_tau_accel_ji = tau_accel_ji_magnitude*particle_component_ratio(self.positions_mat,1,j,i,self.L_star)
                        z_tau_accel_ji = tau_accel_ji_magnitude*particle_component_ratio(self.positions_mat,2,j,i,self.L_star)
                        add_matrix_x_y_z(tau_accel_mat, j, x_tau_accel_ji, y_tau_accel_ji, z_tau_accel_ji)

        self.tau_accel_mat = tau_accel_mat
        self.total_pot_energy = total_pot_energy

    # Calculate position and velocity changes
    def calculate_new_pos_velos(self):
        for i in range(self.num_particles):
            # Find new positions
            r_plus_t_x = 2.*self.positions_mat[i,0] - self.prev_pos_mat[i,0] + self.del_t**2*self.tau_accel_mat[i,0]
            r_plus_t_y = 2.*self.positions_mat[i,1] - self.prev_pos_mat[i,1] + self.del_t**2*self.tau_accel_mat[i,1]
            r_plus_t_z = 2.*self.positions_mat[i,2] - self.prev_pos_mat[i,2] + self.del_t**2*self.tau_accel_mat[i,2]

            # Check for fixes needed on new positions
            r_plus_t_x = position_fix(r_plus_t_x, self.L_star)
            r_plus_t_y = position_fix(r_plus_t_y, self.L_star)
            r_plus_t_z = position_fix(r_plus_t_z, self.L_star)

            # Find new tau_velos
            v_plus_t_x = boundary_cond_dist(r_plus_t_x, self.prev_pos_mat[i,0], self.L_star)/(2.*self.del_t)
            v_plus_t_y = boundary_cond_dist(r_plus_t_y, self.prev_pos_mat[i,1], self.L_star)/(2.*self.del_t)
            v_plus_t_z = boundary_cond_dist(r_plus_t_z, self.prev_pos_mat[i,2], self.L_star)/(2.*self.del_t)

            # Prepare for setting previous positions
            r_current_x = self.positions_mat[i,0]
            r_current_y = self.positions_mat[i,1]
            r_current_z = self.positions_mat[i,2]

            set_matrix_x_y_z(self.prev_pos_mat, i, r_current_x, r_current_y, r_current_z)
            set_matrix_x_y_z(self.positions_mat, i, r_plus_t_x, r_plus_t_y, r_plus_t_z)
            set_matrix_x_y_z(self.tau_velo_mat, i, v_plus_t_x, v_plus_t_y, v_plus_t_z)

    # First time position and velocity changes
    def calculate_1st_new_pos_velos(self):
        for i in range(self.num_particles):
            # Find new positions
            r_plus_t_x = self.positions_mat[i,0] + self.del_t*self.tau_velo_mat[i,0] + self.del_t**2./2.*self.tau_accel_mat[i,0]
            r_plus_t_y = self.positions_mat[i,1] + self.del_t*self.tau_velo_mat[i,1] + self.del_t**2./2.*self.tau_accel_mat[i,1]
            r_plus_t_z = self.positions_mat[i,2] + self.del_t*self.tau_velo_mat[i,2] + self.del_t**2./2.*self.tau_accel_mat[i,2]

            # Check for fixes needed on new positions
            r_plus_t_x = position_fix(r_plus_t_x, self.L_star)
            r_plus_t_y = position_fix(r_plus_t_y, self.L_star)
            r_plus_t_z = position_fix(r_plus_t_z, self.L_star)

            # Find new tau_velos
            v_plus_t_x = self.tau_velo_mat[i,0]  + self.del_t*self.tau_accel_mat[i,0]
            v_plus_t_y = self.tau_velo_mat[i,1]  + self.del_t*self.tau_accel_mat[i,1]
            v_plus_t_z = self.tau_velo_mat[i,2]  + self.del_t*self.tau_accel_mat[i,2]

            set_matrix_x_y_z(self.positions_mat, i, r_plus_t_x, r_plus_t_y, r_plus_t_z)
            set_matrix_x_y_z(self.tau_velo_mat, i, v_plus_t_x, v_plus_t_y, v_plus_t_z)

    # Calculate total energies (total energy should be pretty much the same every step)
    def calculate_total_energies(self):
        self.total_kin_energy = find_kin_energy(self.tau_velo_mat)
        self.energy_per_particle = self.total_kin_energy/self.num_particles + self.total_pot_energy/self.num_particles

    # Calculate total energies (total energy should be pretty much the same every step)
    def calculate_1st_total_energies(self):
        total_pot_energy = 0
        for i in range(self.num_particles):
            for j in range(self.num_particles):
                if(i != j):
                    r_ij = particles_distance(self.positions_mat,i,j,self.L_star)
                    if(r_ij > self.r_c_star):
                        pass
                    else:
                        total_pot_energy += 0.5 * ((1./r_ij)**12. - (1./r_ij)**6.)
        self.total_kin_energy = find_kin_energy(self.tau_velo_mat)
        self.total_pot_energy = total_pot_energy
        self.energy_per_particle = self.total_kin_energy/self.num_particles + self.total_pot_energy/self.num_particles

    # To drive engine for some number of time steps
    def drive_engine(self, steps):
        for i in range(steps):
            print("step: "+str(i))
            self.calculate_tau_accels_pot_energy()
            self.calculate_new_pos_velos()
            if(i<500):
                # Fix the tau_velos for current temperature
                self.fix_tau_velos()
            self.calculate_total_energies()

            self.posfile.write("NEW\n")
            self.forcefile.write("NEW\n")
            for i in range(self.num_particles):
                self.posfile.write(str(self.positions_mat[i,0]))
                self.posfile.write(",")
                self.posfile.write(str(self.positions_mat[i,1]))
                self.posfile.write(",")
                self.posfile.write(str(self.positions_mat[i,2]))
                self.posfile.write("\n")
                self.forcefile.write(str(self.tau_accel_mat[i,0]))
                self.forcefile.write(",")
                self.forcefile.write(str(self.tau_accel_mat[i,1]))
                self.forcefile.write(",")
                self.forcefile.write(str(self.tau_accel_mat[i,2]))
                self.forcefile.write("\n")

            if(self.has_ofile):
                self.ofile.write(str(self.total_kin_energy/self.num_particles))
                self.ofile.write(",")
                self.ofile.write(str(self.total_pot_energy/self.num_particles))
                self.ofile.write(",")
                self.ofile.write(str(self.energy_per_particle))
                self.ofile.write(",")
                self.ofile.write(str((2./3)*find_kin_energy(self.tau_velo_mat)/(self.num_particles)))
                self.ofile.write("\n")

# Actual running of program
if(len(sys.argv) != 4):
    print("Usage: \n\t")
    print(sys.argv[0] + "[number of particles] [number of steps] [energies outfile name]")
    exit()
else:
    my_engine = MDEngine(int(sys.argv[1]), sys.argv[3])
    my_engine.drive_engine(int(sys.argv[2]))