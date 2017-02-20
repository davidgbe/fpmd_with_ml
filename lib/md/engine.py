import numpy as np
from numpy.linalg import norm as mag

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
def particles_distance(positions_mat, particle1, particle2, length):
    x_diff = np.abs(positions_mat[particle1,0]-positions_mat[particle2,0])
    x_diff = np.where(x_diff < 0.5*length, x_diff, length-x_diff)
    
    y_diff = np.abs(positions_mat[particle1,1]-positions_mat[particle2,1])
    y_diff = np.where(y_diff < 0.5*length, y_diff, length-y_diff)
    
    z_diff = np.abs(positions_mat[particle1,2]-positions_mat[particle2,2])
    z_diff = np.where(z_diff < 0.5*length, z_diff, length-z_diff)
    
    return np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

# To get x, y, or z component of distance between two particles, over the magnitude of their total distance
def particle_component_ratio(positions_mat, component, particle1, particle2, length):
    total_distance = particles_distance(positions_mat, particle1, particle2, length)
    component_distance = positions_mat[particle1,component]-positions_mat[particle2,component]
    
    if(component_distance > 0.5*length):
        component_distance = component_distance - length
    elif (component_distance < -0.5*length):
        component_distance = component_distance + length
    if(total_distance == 0):
        return 0
    
    return component_distance/total_distance

# To get the total kinetic energy
def find_kin_energy(velocities_mat, particle_mass):
    return 0.5*particle_mass*np.sum([ mag(velocities_mat[i])**2.0 for i in range(velocities_mat.shape[0]) ])

#To figure out if the particle is on the corner of a cube
def on_corner(positions_mat, particle_num):
    x = positions_mat[particle_num,0]
    y = positions_mat[particle_num,1]
    z = positions_mat[particle_num,2]
    if(x%0.2 == 0 and y%0.2 == 0 and z%0.2 == 0):
        return True
    return False   

# The engine!
class MDEngine:
    # Initialization
    def __init__(self, temp=120.0, num_particles=500, length=0.00001, position_file=None):

        ## Constants/values for this engine ##
        self.tau_0 = 10.0**-13
        self.k_b = 1.38064852*(10**(-23))
        self.temp = temp
        self.num_particles = num_particles
        self.length = length
        self.epsilon = self.temp*self.k_b
        self.omega = self.length**3.0
        self.r_0 = (self.omega*3.0/(4.0*np.pi*self.num_particles))**(1.0/3.0)
        self.sigma = (0.8*self.omega/self.num_particles)**(1.0/3.0)
        self.r_c = 2.5*self.sigma
        self.particle_mass = (self.tau_0**2.0)*self.epsilon/(self.r_0**2.0)
        self.del_t = self.tau_0*(10**-2)

        if position_file is None:
            # generate positions
            self.generate_positions()
        else:
            # get positions from position file
            pass
        self.generate_velocities()

        self.calculate_total_energies()
        # print(self.total_kin_energy)
        # print(self.total_pot_energy)
        print(self.energy_per_particle)

        self.calculate_forces()
        self.calculate_1st_new_pos_velos()
        self.calculate_total_energies()
        print("kinetic")
        print(self.total_kin_energy)
        print("potential")
        print(self.total_pot_energy)
        print(self.energy_per_particle)

    # Generate face centered cubic positions for particles
    def generate_positions(self):
        num_cells_per_edge = int(round((self.num_particles/4)**(1./3)))
        cell_length = self.length/(num_cells_per_edge)
        positions_mat = np.zeros((self.num_particles, 3.0))
        #orig_pos_mat = np.zeros((self.num_particles, 3.0))
        prev_pos_mat = np.zeros((self.num_particles, 3.0))
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

    # Generate random velocities for each particle, and fix such that total temperature is correct
    def generate_velocities(self):
        velocities_mat = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            for j in range(3):
                velocities_mat[i,j] = rand_box_muller()

        total_kin_energy = find_kin_energy(velocities_mat, self.particle_mass)

        temp_prime = (2.0/3.0)*total_kin_energy/(self.num_particles*self.k_b)
        velocity_fix = np.sqrt(self.temp/temp_prime)

        for i in range(self.num_particles):
            for j in range(3):
                velocities_mat[i,j] = velocities_mat[i,j]*velocity_fix

        self.velocities_mat = velocities_mat


    # Calculate the potential energy of each particle, and through this the force on that particle
    def calculate_forces(self):

        forces_mat = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            for j in range(self.num_particles):
                if(i != j):
                    r_ij = particles_distance(self.positions_mat,i,j,self.length)
                    if(r_ij > self.r_c):
                        pass
                    else:
                        force_ij_signed_magnitude = -4.*self.epsilon*(-12.*self.sigma**12./(r_ij**13.) + 6.*self.sigma**6./(r_ij**7.))
                        # Component forces
                        x_force_ij = force_ij_signed_magnitude*particle_component_ratio(self.positions_mat,0,i,j,self.length)
                        y_force_ij = force_ij_signed_magnitude*particle_component_ratio(self.positions_mat,1,i,j,self.length)
                        z_force_ij = force_ij_signed_magnitude*particle_component_ratio(self.positions_mat,2,i,j,self.length)
                        add_matrix_x_y_z(forces_mat, i, x_force_ij, y_force_ij, z_force_ij)
        self.forces_mat = forces_mat

    # Calculate position and velocity changes
    def calculate_new_pos_velos(self):
        for i in range(self.num_particles):
            r_plus_t_x = self.positions_mat[i,0]*2 - self.prev_pos_mat[i,0] + self.del_t**2*self.forces_mat[i,0]/self.particle_mass
            r_plus_t_y = self.positions_mat[i,1]*2 - self.prev_pos_mat[i,1] + self.del_t**2*self.forces_mat[i,1]/self.particle_mass
            r_plus_t_z = self.positions_mat[i,2]*2 - self.prev_pos_mat[i,2] + self.del_t**2*self.forces_mat[i,2]/self.particle_mass

            v_plus_t_x = (r_plus_t_x-self.prev_pos_mat[i,0])/(2*self.del_t)
            v_plus_t_y = (r_plus_t_y-self.prev_pos_mat[i,1])/(2*self.del_t)
            v_plus_t_z = (r_plus_t_z-self.prev_pos_mat[i,2])/(2*self.del_t)

            r_current_x = self.positions_mat[i,0]
            r_current_y = self.positions_mat[i,1]
            r_current_z = self.positions_mat[i,2]

            set_matrix_x_y_z(self.prev_pos_mat, i, r_current_x, r_current_y, r_current_z)
            set_matrix_x_y_z(self.positions_mat, i, r_plus_t_x, r_plus_t_y, r_plus_t_z)
            set_matrix_x_y_z(self.velocities_mat, i, v_plus_t_x, v_plus_t_y, v_plus_t_z)

    # First time position and velocity changes
    def calculate_1st_new_pos_velos(self):
        for i in range(self.num_particles):
            r_plus_t_x = self.positions_mat[i,0] + self.del_t*self.velocities_mat[i,0] + self.del_t**2./2.*self.forces_mat[i,0]/(self.particle_mass)
            r_plus_t_y = self.positions_mat[i,1] + self.del_t*self.velocities_mat[i,1] + self.del_t**2./2.*self.forces_mat[i,1]/(self.particle_mass)
            r_plus_t_z = self.positions_mat[i,2] + self.del_t*self.velocities_mat[i,2] + self.del_t**2./2.*self.forces_mat[i,2]/(self.particle_mass)

            v_plus_t_x = self.velocities_mat[i,0]  + self.del_t*self.forces_mat[i,0]/(self.particle_mass)
            v_plus_t_y = self.velocities_mat[i,1]  + self.del_t*self.forces_mat[i,1]/(self.particle_mass)
            v_plus_t_z = self.velocities_mat[i,2]  + self.del_t*self.forces_mat[i,2]/(self.particle_mass)

            #set_matrix_x_y_z(self.prev_pos_mat, i, self.positions_mat[i,0], self.positions_mat[i,1], self.positions_mat[i,2])
            set_matrix_x_y_z(self.positions_mat, i, r_plus_t_x, r_plus_t_y, r_plus_t_z)
            set_matrix_x_y_z(self.velocities_mat, i, v_plus_t_x, v_plus_t_y, v_plus_t_z)

    # Calculate total energies (should be pretty much the same every step)
    def calculate_total_energies(self):
        total_pot_energy = 0
        for i in range(self.num_particles):
            for j in range(self.num_particles):
                if(i != j):
                    r_ij = particles_distance(self.positions_mat,i,j,self.length)
                    if(r_ij > self.r_c):
                        pass
                    else:
                        total_pot_energy += 4*self.epsilon*((self.sigma/r_ij)**12 - (self.sigma/r_ij)**6)
        self.total_kin_energy = find_kin_energy(self.velocities_mat, self.particle_mass)
        self.total_pot_energy = total_pot_energy
        self.energy_per_particle = self.total_kin_energy/self.num_particles + self.total_pot_energy/self.num_particles

    # To drive engine for some number of time steps
    def drive_engine(self, steps):
        for i in range(steps-1):
            self.calculate_forces()
            self.calculate_new_pos_velos()
            self.calculate_total_energies()
            print("kinetic")
            print(self.total_kin_energy/self.num_particles)
            print("potential")
            print(self.total_pot_energy/self.num_particles)
            print(self.energy_per_particle)

            # for i in range(self.num_particles):
            #     x_diff = np.abs(self.positions_mat[i,0]-self.orig_pos_mat[i,0])
            #     x_diff = np.where(x_diff < 0.5*self.length, x_diff, self.length-x_diff)
            #     y_diff = np.abs(self.positions_mat[i,1]-self.orig_pos_mat[i,1])
            #     y_diff = np.where(y_diff < 0.5*self.length, y_diff, self.length-y_diff)
            #     z_diff = np.abs(self.positions_mat[i,2]-self.orig_pos_mat[i,2])
            #     z_diff = np.where(z_diff < 0.5*self.length, z_diff, self.length-z_diff)
            #     if(np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)>0.00002):
            #         print("yuhyhyhyutu")


my_engine = MDEngine()
my_engine.drive_engine(20)