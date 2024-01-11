# Importar bibliotecas
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from deap import base, creator, tools, algorithms
# from scipy.optimize import minimize

# Importar classes personalizadas de quadcopter
from QuadcopterSTSM import QuadcopterSim

# Tempo de execução da simulação
sim_start = 0  # tempo inicial da simulação
sim_end = 22 # tempo final da simulação em segundos
dt = 0.01  # tamanho do passo em segundos
time_index = np.arange(sim_start, sim_end + dt, dt)

# Condições iniciais
# posição desejada [x, y, z] no referencial inercial - metros
p_ref = np.array([5., -5, -1], dtype = float).reshape(3,1)

# localização inicial [x, y, z] no referencial inercial - metros
pos = np.array([5, -5., 1], dtype = float).reshape(3,1)
# velocidade inicial [x; y; z] no referencial inercial - m/s
vel = np.array([0., 0., 0], dtype = float).reshape(3,1)
# ângulos de Euler iniciais [phi, theta, psi] em relação ao referencial inercial em graus
ang = np.array([0., 0., 0.], dtype = float).reshape(3,1)

# Adicionar perturbações iniciais aleatórias de rolagem, arfagem e guinada
# deviation = 10  # magnitude da perturbação inicial em graus/s
random_set =   np.random.rand(3) # np.array([0.5, 0.5, 0.9], dtype = float) #
# velocidade angular inicial [phi_dot, theta_dot, psi_dot]
# ang_vel = np.deg2rad(2 * deviation * random_set - deviation)
# print(ang_vel)
ang_vel = np.array([0.000169077, 0.000169077, -0.0264164])
# ang_vel = np.array([0, 0, -0.0264164])
# ang_vel = np.array([0, 0, 0])
# ang_vel = np.array([-0.31680387, -0.10896892, -0.23575788])

ang_vel = np.array(ang_vel,  dtype = float).reshape(3,1)
ang_vel_init = ang_vel.copy()  # registro para exibição posterior

gravity = 9.81  # aceleração devido à gravidade, m/s^2

# Criar quadcopter com objetos de controlador de posição e ângulo
quadcopter = QuadcopterSim(p_ref, pos, vel, ang, ang_vel, dt)


# Inicialização de arrays de resultados
total_error = []
position_total = []
total_thrust = []
melhor_score = 100
saida_altitude = []
erro_altitude = []

position = [np.empty(0), np.empty(0), np.empty(0)]
velocity = [np.empty(0), np.empty(0), np.empty(0)]
angle = [np.empty(0), np.empty(0), np.empty(0)]
angle_vel = [np.empty(0), np.empty(0), np.empty(0)]
motor_thrust = [np.empty(0) for _ in range(4)]
body_torque = [np.empty(0) for _ in range(3)]
motor_speed = [np.empty(0) for _ in range(4)]
saida_atitude = [np.empty(0) for _ in range(3)]
erro_atitude = [np.empty(0) for _ in range(3)]

# Loop de simulação
for time, t_index in enumerate(time_index):
    # quadcopter.stepSTSM()
    # quadcopter.stepMA()
    quadcopter.stepPID()
    
    position_total.append(np.linalg.norm(quadcopter.pos))
    saida_altitude = np.append(saida_altitude, quadcopter.sAltitude)
    erro_altitude = np.append(erro_altitude, (quadcopter.erro_altitude))
    
    for i in range(3):
        position[i] = np.append(position[i], quadcopter.pos[i])
        velocity[i] = np.append(velocity[i], quadcopter.vel[i])
        angle[i] = np.append(angle[i], (quadcopter.angle[i]))
        angle_vel[i] = np.append(angle_vel[i], (quadcopter.ang_vel[i]))
        saida_atitude[i] = np.append(saida_atitude[i], quadcopter.sAtitude[i])
        erro_atitude[i] = np.append(erro_atitude[i], (quadcopter.erro_atitude[i]))
    # print(f"posição no main:{position[2]}")

    for i in range(4):
        motor_thrust[i] = np.append(motor_thrust[i], quadcopter.speeds[i] * quadcopter.kT)
        motor_speed[i] = np.append(motor_speed[i], quadcopter.speeds[i])

    total_thrust.append(quadcopter.kT * np.sum(quadcopter.speeds))
    total_error.append(np.linalg.norm(quadcopter.prev_error))

    if quadcopter.done:
        break




def write_init_ang_vel_to_screen():
    ''' 
    The program initializes with a random perturbation in angular velocity on the vehicle. 
    This simulates a wind disturbace.
    This is a display of the random disturbance
    '''
    print('Initial angular velocities (rad/s):')
    print(np.rad2deg(ang_vel_init))
    print('Total magnitude of angular velocity (rad/s)')
    print(np.linalg.norm(np.rad2deg(ang_vel_init)))
    

def error_plot():
    ''' Plots to the magnitude of the position error vector (m)'''
    plt.plot(time_index[:len(total_error)], total_error)
    plt.title('Quadcopter distance from reference point over time')
    plt.xlabel('time (s)')
    plt.ylabel('error (m)')
    plt.show()


def simple_plot():
    ''' 
    Plots the laterial position, vertical position, and Euler angles over time.
    This is a quick plot for trouble shooting
    '''
    fig = plt.figure(num=None, figsize=(10, 6), dpi=80,
                     facecolor='w', edgecolor='k')
    # Lateral position plots
    axes = fig.add_subplot(1, 3, 1)
    # axes.plot(time_index[:len(position[0])], position[0], label= 'x')
    # axes.plot(time_index[:len(position[1])], position[1], label= 'y')
    axes.plot(position[0], position[1])
    axes.set_title('Lateral Postion Over Time')
    axes.set_xlabel('x-position (m)')
    axes.set_ylabel('y-position (m)')
    axes.legend()

    # Vertical position plot
    axes = fig.add_subplot(1, 3, 2)
    axes.plot(time_index[:len(position[2])], position[2], label='z')
    axes.set_title('Vertical Position Over Time')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('altitude (m)')

    # Angles over time
    axes = fig.add_subplot(1, 3, 3)
    axes.plot(time_index[:len(angle[0])], angle[0], label='phi')
    axes.plot(time_index[:len(angle[1])], angle[1], label='theta')
    axes.plot(time_index[:len(angle[2])], angle[2], label='psi')
    axes.set_title('Euler Angles Over Time')
    axes.set_xlabel('time (s)')
    axes.set_ylabel('angle (rad)')
    axes.legend()

    plt.tight_layout(pad=0.4, w_pad=2.5, h_pad=2.0)
    plt.show()


def plot_pos():
    """ Função que plota as tres posições ao decorrer do tempo em que a simulação foi simulada """
    fig = plt.figure(num=None, figsize=(10, 6), dpi=80,
                     facecolor='w', edgecolor='k')

    # Plotting x-position
    axes = fig.add_subplot(1, 3, 1)
    axes.plot(time_index[:len(position[0])], position[0])
    axes.set_title('X-Position Over Time')
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('X-Position (m)')

    # Plotting y-position
    axes = fig.add_subplot(1, 3, 2)
    axes.plot(time_index[:len(position[1])], position[1])
    axes.set_title('Y-Position Over Time')
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('Y-Position (m)')

    # Plotting z-position
    axes = fig.add_subplot(1, 3, 3)
    axes.plot(time_index[:len(position[2])], position[2])
    axes.set_title('Z-Position Over Time')
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('Z-Position (m)')

    plt.tight_layout(pad=0.4, w_pad=2.5, h_pad=2.0)
    plt.show()


def total_plot():
    '''
    Este é um gráfico completo dos resultados. Ele irá plotar o caminho de voo em 3D, posições vertical e lateral,
    velocidade lateral, empuxo dos motores, torques do corpo, ângulos de Euler e velocidade angular do veículo.
    '''
    fig = plt.figure(num=None, figsize=(16, 8), dpi=80,
                     facecolor='w', edgecolor='k')

    # 3D Flight path
    axes = fig.add_subplot(2, 4, 1, projection='3d')
    axes.plot(position[0], position[1], position[2])
    axes.set_title('Caminho de Voo')
    axes.set_xlabel('x (m)')
    axes.set_ylabel('y (m)')
    axes.set_zlabel('z (m)')


    # Lateral position plots
    axes = fig.add_subplot(2, 4, 2)
    axes.plot(time_index[:len(position[0])], position[0], label='x')
    axes.plot(time_index[:len(position[1])], position[1], label='y')
    axes.set_title('Posição Lateral')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('posição (m)')
    axes.legend()

    # Vertical position plot
    axes = fig.add_subplot(2, 4, 3)
    axes.plot(time_index[:len(position[2])], position[2], label='z')
    axes.set_title('Posição Vertical')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('altitude (m)')

    # Lateral velocity plots
    axes = fig.add_subplot(2, 4, 4)
    axes.plot(time_index[:len(velocity[0])], velocity[0], label='d(x)/dt')
    axes.plot(time_index[:len(velocity[1])], velocity[1], label='d(y)/dt')
    axes.plot(time_index[:len(velocity[2])], velocity[2], label='d(z)/dt')
    axes.set_title('Velocidade Linear')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('velocidade (m/s)')
    axes.legend()
    
    # Saida controlador de atitude
    axes = fig.add_subplot(2, 4, 5)
    axes.plot(time_index[:len(saida_atitude[0])],
              saida_atitude[0], label='phi')
    axes.plot(time_index[:len(saida_atitude[1])],
              saida_atitude[1], label='theta')
    axes.plot(time_index[:len(saida_atitude[2])],
              saida_atitude[2], label='psi')
    axes.set_title('Controlador de Atitude')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel(' Torque Desejado(N.m)')
    axes.legend()
    
    # # Motor speed plots
    # axes = fig.add_subplot(2, 4, 5)
    # axes.plot(time_index[:len(motor_thrust[0])],
    #           motor_thrust[0], label='motor 1')
    # axes.plot(time_index[:len(motor_thrust[1])],
    #           motor_thrust[1], label='motor 2')
    # axes.plot(time_index[:len(motor_thrust[2])],
    #           motor_thrust[2], label='motor 3')
    # axes.plot(time_index[:len(motor_thrust[3])],
    #           motor_thrust[3], label='motor 4')
    # axes.set_title('Empuxo dos Motores')
    # axes.set_xlabel('tempo (s)')
    # axes.set_ylabel('Empuxo dos Motores (N)')
    # axes.legend()

    # speeds plot
    # axes = fig.add_subplot(2, 4, 6)
    # axes.plot(time_index[:len(motor_speed[0])],
    #           motor_speed[0], label='motor 1')
    # axes.plot(time_index[:len(motor_speed[1])],
    #           motor_speed[1], label='motor 2')
    # axes.plot(time_index[:len(motor_speed[2])],
    #           motor_speed[2], label='motor 3')
    # axes.plot(time_index[:len(motor_speed[3])],
    #           motor_speed[3], label='motor 4')
    # axes.set_title('Velocidades dos Motores')
    # axes.set_xlabel('tempo (s)')
    # axes.set_ylabel('Velocidade dos Motores')
    # axes.legend()
    
    # Erro controle de atitude
    axes = fig.add_subplot(2, 4, 6)
    axes.plot(time_index[:len(erro_atitude[0])],
              erro_atitude[0], label='phi')
    axes.plot(time_index[:len(erro_atitude[1])],
              erro_atitude[1], label='theta')
    axes.plot(time_index[:len(erro_atitude[2])],
              erro_atitude[2], label='psi')
    axes.set_title('Erro de Atitude')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel(' Torque Desejado(N.m)')
    axes.legend()

    # Angles over time
    axes = fig.add_subplot(2, 4, 7)
    axes.plot(time_index[:len(angle[0])], angle[0], label='phi')
    axes.plot(time_index[:len(angle[1])], angle[1], label='theta')
    axes.plot(time_index[:len(angle[2])], angle[2], label='psi')
    axes.set_title('Ângulos de Euler')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('ângulo (rad)')
    axes.legend()

    # Angular velocity over time
    axes = fig.add_subplot(2, 4, 8)
    axes.plot(time_index[:len(angle_vel[0])], angle_vel[0], label='d(phi)/dt')
    axes.plot(time_index[:len(angle_vel[1])],
              angle_vel[1], label='d(theta)/dt')
    axes.plot(time_index[:len(angle_vel[2])], angle_vel[2], label='d(psi)/dt')
    axes.set_title('Velocidade Angular')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('velocidade angular (rad/s)')
    axes.legend()

    plt.tight_layout(pad=0.4, w_pad=2.5, h_pad=2.0)
    plt.show()

def altitude_plot():
    
    fig = plt.figure(num=None, figsize=(8, 8), dpi=80,
                     facecolor='w', edgecolor='k')

    # Saida do controlador Altitude
    axes = fig.add_subplot(2, 2, 1)
    axes.plot(time_index[:len(saida_altitude)], saida_altitude, label='N')
    axes.set_title('Controlador de Altitude')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('thust desejado (N)')


    # Erro controle de altitude
    axes = fig.add_subplot(2, 2, 2)
    axes.plot(time_index[:len(erro_altitude)], erro_altitude, label='erro')
    axes.set_title('Erro de Altitude')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('erro (m)')

    # Vertical position plot
    axes = fig.add_subplot(2, 2, 3)
    axes.plot(time_index[:len(position[2])], position[2], label='z')
    axes.set_title('Posição Vertical')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('altitude (m)')

    # Lateral velocity plots
    axes = fig.add_subplot(2, 2, 4)
    # axes.plot(time_index[:len(velocity[0])], velocity[0], label='d(x)/dt')
    # axes.plot(time_index[:len(velocity[1])], velocity[1], label='d(y)/dt')
    axes.plot(time_index[:len(velocity[2])], velocity[2], label='d(z)/dt')
    axes.set_title('Velocidade Linear')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('velocidade (m/s)')
    axes.legend()

    plt.tight_layout(pad=0.4, w_pad=2.5, h_pad=2.0)
    plt.show()
    
def atitude_plot():
    
    fig = plt.figure(num=None, figsize=(8, 8), dpi=80,
                     facecolor='w', edgecolor='k')

    # Saida controlador de atitude
    axes = fig.add_subplot(2, 2, 1)
    axes.plot(time_index[:len(saida_atitude[0])],
              saida_atitude[0], label='phi')
    axes.plot(time_index[:len(saida_atitude[1])],
              saida_atitude[1], label='theta')
    axes.plot(time_index[:len(saida_atitude[2])],
              saida_atitude[2], label='psi')
    axes.set_title('Controlador de Atitude')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel(' torque desejado(N.m)')
    axes.legend()
        
    # Erro controle de atitude
    axes = fig.add_subplot(2, 2, 2)
    axes.plot(time_index[:len(erro_atitude[0])],
              erro_atitude[0], label='phi')
    axes.plot(time_index[:len(erro_atitude[1])],
              erro_atitude[1], label='theta')
    axes.plot(time_index[:len(erro_atitude[2])],
              erro_atitude[2], label='psi')
    axes.set_title('Erro de Atitude')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('erro (rad)')
    axes.legend()

    # Angles over time
    axes = fig.add_subplot(2, 2, 3)
    axes.plot(time_index[:len(angle[0])], angle[0], label='phi')
    axes.plot(time_index[:len(angle[1])], angle[1], label='theta')
    axes.plot(time_index[:len(angle[2])], angle[2], label='psi')
    axes.set_title('Ângulos de Euler')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('ângulo (rad)')
    axes.legend()

    # Angular velocity over time
    axes = fig.add_subplot(2, 2, 4)
    axes.plot(time_index[:len(angle_vel[0])], angle_vel[0], label='d(phi)/dt')
    axes.plot(time_index[:len(angle_vel[1])],
              angle_vel[1], label='d(theta)/dt')
    axes.plot(time_index[:len(angle_vel[2])], angle_vel[2], label='d(psi)/dt')
    axes.set_title('Velocidade Angular')
    axes.set_xlabel('tempo (s)')
    axes.set_ylabel('velocidade angular (rad/s)')
    axes.legend()

    plt.tight_layout(pad=0.4, w_pad=2.5, h_pad=2.0)
    plt.show()
# write_init_ang_vel_to_screen()
# plot_pos()
# error_plot()
# simple_plot()
altitude_plot()
atitude_plot()
# total_plot()
