import numpy as np


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
def rand_box_muller():
    u1 = np.random.random()
    u2 = np.random.random()
    t = np.sqrt((-2) * np.log(u1))
    v = 2 * np.pi * u2
    return t * np.cos(v)

# To get the magnitude of the distance between two particles
def particles_distance(positions_mat, particle1, particle2):
    x_diff = positions_mat[particle1,0]-positions_mat[particle2,0]
    y_diff = positions_mat[particle1,1]-positions_mat[particle2,1]
    z_diff = positions_mat[particle1,2]-positions_mat[particle2,2]
    return np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

# To get x, y, or z component of distance between two particles, over the magnitude of their total distance
def particle_component_distance(positions_mat, component, particle1, particle2):
    total_distance = particles_distance(positions_mat, particle1, particle2)
    component_distance = positions_mat[particle1,component]-positions_mat[particle2,component]
    return total_distance/component_distance

# To get the magnitude of a particle's velocity
def particle_velocity(velocities_mat, particle_num):
    x_velocity = velocities_mat[particle_num,0]
    y_velocity = velocities_mat[particle_num,1]
    z_velocity = velocities_mat[particle_num,2]
    return np.sqrt(x_velocity**2 + y_velocity**2 + z_velocity**2)

# Gonna be for something...
def integration_algorithm():
    return 1

# The engine!
class MDEngine:

    # Initialization
    def __init__(self, temp=120, position_file=None):
        self.temp = temp

        if position_file is None:
            # generate positions
            self.generate_positions()
        else:
            # get positions from position file
            print("else")
        self.generate_velocities()

    # Generate face centered cubic positions for particles
    def generate_positions(self, num_cells_per_edge=5, length=1.0):
        num_cells = num_cells_per_edge**3
        cell_length = length/(num_cells_per_edge)
        num_particles = 4*num_cells
        positions_mat = np.zeros((num_particles, 3))
        i = 0;

        # add all cell-corner particles (should be first num_cells^3 particles)
        for j, k, l in triple_for(num_cells_per_edge):
            set_matrix_x_y_z(positions_mat, i, j*cell_length, k*cell_length, l*cell_length)
            i += 1

        # add all face particles with n*cell_length x values
        for j, k, l in triple_for(num_cells_per_edge):
            set_matrix_x_y_z(positions_mat, i, j*cell_length, (k+(np.sqrt(2)/2))*cell_length, (l+(np.sqrt(2)/2))*cell_length)
            i += 1

        # add all face particles with n*cell_length y values
        for j, k, l in triple_for(num_cells_per_edge):
            set_matrix_x_y_z(positions_mat, i, (j+(np.sqrt(2)/2))*cell_length, k*cell_length, (l+(np.sqrt(2)/2))*cell_length)
            i += 1

        # add all face particles with n*cell_length z values
        for j, k, l in triple_for(num_cells_per_edge):
            set_matrix_x_y_z(positions_mat, i, (j+(np.sqrt(2)/2))*cell_length, (k+(np.sqrt(2)/2))*cell_length, l*cell_length)
            i += 1
            
        self.positions_mat = positions_mat

    # Generate random velocities for each particle, and fix such that total temperature is correct
    def generate_velocities(self, num_particles=500, particle_mass=1.0):
        velocities_mat = np.zeros((num_particles, 3))
        for i in range(num_particles):
            for j in range(3):
                velocities_mat[i,j] = rand_box_muller()

        total_kin_energy = 0
        for i in range(num_particles):
            total_kin_energy += (1/2)*particle_mass*(particle_velocity(velocities_mat,i)**2)

        temp_prime = (2/3)/(num_particles)*total_kin_energy#/kb...what is kb?
        velocity_fix = np.sqrt(temp_prime/self.temp)
        for i in range(num_particles):
            for j in range(3):
                velocities_mat[i,j] = velocities_mat[i,j]*velocity_fix
        self.velocities_mat = velocities_mat

    # Calculate the potential energy of each particle, and through this the force on that particle
    def calculate_forces(self, num_particles=500, length=1):
        omega = length^3
        epsilon = 1 # ?!?!? what is this... probably not actually 1
        sigma = (0.8*omega/num_particles)**(1/3)
        r_c = 2.5*sigma

        total_potential_energy = 0
        potential_mat = np.zeros((num_particles, num_particles))
        forces_mat = np.zeros((num_particles, 3))
        for i in range(num_particles):
            for j in range(num_particles):
                if(i != j):
                    r_ij = particles_distance(self.positions_mat,i,j)
                    if(r_ij > r_c):
                        pass
                        #potential_mat[i,1] = 0 # Not sure what Kalia wrote here, operating off of wikipedia
                    else:
                        # Again, following wikipedia here
                        potential_mat[i,j] += 4*epsilon*((sigma/r_ij)**12 - (sigma/r_ij)**6) - (4*epsilon*((sigma/r_c)**12 - (sigma/r_c)**6))
                        total_potential_energy += potential_mat[i,j]
                        # forces_mat[i,j] += -4*epsilon*(-12*sigma**12/(r_ij**13) + 6*sigma**6/(r_ij**7)) # Negative derivative of potential
                        # x_force_ij = 
                        # y_force_ij = 
                        # z_force_ij = 
                        # add_matrix_x_y_z(forces_mat, i, x_force_ij, y_force_ij, z_force_ij)

my_engine = MDEngine()
my_engine.calculate_forces()