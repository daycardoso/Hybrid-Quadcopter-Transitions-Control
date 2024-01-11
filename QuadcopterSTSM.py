import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import copy
import pprint
import numpy as np
import math

def sat(delta,x):
    if x < -delta or x == -delta:
        x = -1
    elif abs(x) < delta and delta > 0:
        x = x/delta
    elif x > delta or x == delta:
        x = 1
    
    return x

def c(x):
    ''' cosine of angle in radians'''
    return np.cos(x)


def s(x):
    ''' sine of angle in radians'''
    return np.sin(x)


def t(x):
    ''' tangent of angle in radians'''
    return np.tan(x)


class Quadcopter():

    def __init__(self, pos, vel, angle, ang_vel, dt):
        """
        Inicializa o objeto Quadcopter.

        Parâmetros:
        - pos (lista): A posição inicial do quadricóptero no referencial inercial [x, y, z], em metros.
        - vel (lista): A velocidade inicial do quadricóptero no referencial inercial [x_dot, y_dot, z_dot], em m/s.
        - angle (lista): A orientação inicial do quadricóptero no referencial inercial [roll, pitch, yaw] -> [phi, theta, psi], em radianos.
        - ang_vel (lista): A velocidade angular inicial do quadricóptero no referencial inercial [phi_dot, theta_dot, psi_dot], em rad/s.
        - pos_ref (lista): A posição desejada do quadricóptero no referencial inercial [x, y, z], em metros.
        - dt (float): O passo de tempo para a simulação, em segundos.
        """
        # Variáveis de simulação inicial
        self.pos = pos
        self.vel = vel
        self.angle = angle
        self.ang_vel = ang_vel
        self.lin_acc = np.array([0., 0., 0.]).reshape(3, 1)
        self.ang_acc = np.array([0., 0., 0.]).reshape(3, 1)

        # Estados de referência desejados
        self.vel_ref = [0., 0., 0.]
        self.lin_acc_ref = [0., 0., 0.]
        self.angle_ref = [0., 0., 0.]
        self.ang_vel_ref = [0., 0., 0.]
        self.ang_acc_ref = [0., 0., 0.]

        # Gerais
        self.mass = 1.2  # Massa da plataforma (kg)
        self.gravity = 9.81  # Aceleração da gravidade (m/s^2)
        self.num_motors = 4  # número de motores no veículo
        self.r = 0.15  # Raio da hélice (m)
        self.r_esfera = 0.01  # calcular a massa adicionada considerando-a como uma esfera
        self.L = 0.19  # Comprimento do braço (m)
        self.V = 2e-3  # Volume (m^3)
        self.Ixx = 2.365e-2  # Inércia de rotação em torno do eixo X (kg/m^2)
        self.Iyy = 1.318e-2  # Inércia de rotação em torno do eixo Y (kg/m^2)
        self.Izz = 1.318e-2  # Inércia de rotação em torno do eixo Z (kg/m^2)
        # self.I = np.array([[self.Ixx, 0, 0],[0, self.Iyy, 0],[0, 0, self.Izz]])
        self.J = np.diag([self.Ixx, self.Iyy, self.Izz])

        self.speeds = np.zeros(self.num_motors)
        self.motor_torques = [0., 0., 0., 0.]

        # Varivaies iniciais do controle
        self.output = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
        self.prev_phi_error = 0.0
        self.prev_phi_output = 0.0
        self.prev_theta_error = 0.0
        self.prev_theta_output = 0.0
        self.prev_psi_error = 0.0
        self.prev_psi_output = 0.0        


        # Medidas de tempo
        self.time = 0
        self.dt = dt

        self.done = False

    def body2inertial_rotation(self):
        ''' 
        Rotações de Euler do quadro do corpo para o quadro inercial global
        ângulo 0 = roll (eixo x, phi)
        ângulo 1 = pitch (eixo y, theta)
        ângulo 2 = yaw (eixo z, psi)
        '''

        c1 = c(self.angle[0])
        s1 = s(self.angle[0])
        c2 = c(self.angle[1])
        s2 = s(self.angle[1])
        c3 = c(self.angle[2])
        s3 = s(self.angle[2])

        R = np.array([[c2*c3, c3*s1*s2 - c1*s3, s1*s3 + c1*s2*c3],
                      [c2*s3, c1*c3 + s1*s2*s3, c1*s3*s2 - c3*s1],
                      [-s2, c2*s1, c1*c2]])

        return R

    def inertial2body_rotation(self):
        ''' 
        Rotações de Euler do quadro do quadro inercial global para o do corpo 
        ângulo 0 = roll (eixo x, phi)
        ângulo 1 = pitch (eixo y, theta)
        ângulo 2 = yaw (eixo z, psi)
        '''
        c1 = c(self.angle[0])
        s1 = s(self.angle[0])
        c2 = c(self.angle[1])
        s2 = s(self.angle[1])
        c3 = c(self.angle[2])
        s3 = s(self.angle[2])
        # R = np.transpose(self.body2inertial_rotation())
        R = np.array([
        [c3*c2, -c2*s3, s2],
        [c1*s3 + c3*s1*s2, c1*c3 - s1*s2*s3, -c2*s1],
        [-c3*c1*s2 + s1*s3, c1*c3*s2 + s1*s2*s3, c1*c2]
        ])

        return R

    def sat_angle(self):
        # |η| < ηL com limite ηL = [π/2 π/2 2π]^T

        # Define o limite ηL
        # eta_L = np.array([np.pi/2, np.pi/2, 2*np.pi])
        eta_L = np.array([np.pi, np.pi, np.pi])

        # Garante |η| < ηL
        self.angle[0] = np.clip(self.angle[0], -eta_L[0], eta_L[0])
        self.angle[1] = np.clip(self.angle[1], -eta_L[1], eta_L[1])
        self.angle[2] = np.clip(self.angle[2], -eta_L[2], eta_L[2])

        return self.angle
   
    def satcontrol(self, U, flag):
        
        if flag == "ar":
            U = np.array([U[0].item() + self.mass * self.gravity,U[1].item(),U[2].item(),U[3].item()]).reshape(4,1)
            if U[0].item() > 18:
                U[0] = 18
            if U[0].item() < 7:
                U[0] = 7
            sat = 3
            sat2 = 2
            if U[1].item() > sat2:
                U[1] = sat2
            if U[1].item()< -sat2:
                U[1] = -sat2
            if U[2].item() > sat2:
                U[2] = sat2
            if U[2].item() < -sat2:
                U[2] = -sat2
            if U[3].item() > sat:
                U[3] = sat
            if U[3].item() < -sat:
                U[3] = -sat
            self.prev_phi_output = U[1].item()
            # print(self.prev_phi_output)
            self.prev_theta_output = U[2].item()
            self.prev_psi_output = U[3].item()
        
        elif flag == "agua":
            U = np.array([U[0].item() ,U[1].item(),U[2].item(),U[3].item()]).reshape(4,1)
            if U[0].item() > 0:
                U[0] = 0
            if U[0].item() < -21:
                U[0] = -21
            sat = 5
            sat2 = .75
            if U[1] > sat2:
                U[1] = sat2
            if U[1] < -sat2:
                U[1] = -sat2
            if U[2] > sat2:
                U[2] = sat2
            if U[2] < -sat2:
                U[2] = -sat2
            if U[3] > sat:
                U[3] = sat
            if U[3] < -sat:
                U[3] = -sat
            self.prev_phi_output = U[1].item()
            self.prev_theta_output = U[2].item()
            self.prev_psi_output = U[3].item()
            
        elif flag == "transicao":
            U = np.array([U[0].item() + self.mass * self.gravity,U[1].item(),U[2].item(),U[3].item()]).reshape(4,1)
            if U[0].item() > 0:
                U[0] = 0
            if U[0].item() < -21:
                U[0] = -21
            sat = 2.5
            sat2 = .75
            if U[1] > sat2:
                U[1] = sat2
            if U[1] < -sat2:
                U[1] = -sat2
            if U[2] > sat2:
                U[2] = sat2
            if U[2] < -sat2:
                U[2] = -sat2
            if U[3] > sat:
                U[3] = sat
            if U[3] < -sat:
                U[3] = -sat
            self.prev_phi_output = U[1].item()
            self.prev_theta_output = U[2].item()
            self.prev_psi_output = U[3].item()
            
        return U
        
    def des2speeds(self, U):
        # print(f"speeds:{self.speeds}")
        self.speeds = np.sqrt(np.maximum(np.dot(self.K_inv, U) / self.kT , 0))
        print(U[1])
        for i in range(4):
            if self.speeds[i] > self.maxT:
                self.speeds[i] = self.maxT

            # if self.speeds[i] < self.minT:
            #     self.speeds[i] = self.minT
        
        return self.speeds

    def controlePos(self, ref, kp, ki, kd, ts, atual, flag):
             
        error = ref - atual        
        # print(f"erro:{error}")
        integral = error * ts
        derivative = (error - self.prev_error) / ts
        #print(f"I:{integral}")
        output = (self.prev_output + kp * ( error - self.prev_error) + ki * integral + kd * derivative)
        output = output.item()
        #output = error * kp
        if flag == "posAr":
            if output < 7:
                output = 7
            elif output > 18:
                output = 18
            self.prev_output = output
            self.prev_error = error
                
        elif flag == "posAgua":
            if output < -21:
                output = -21
            elif output > 0:
                output = 0
            self.prev_output = output
            self.prev_error = error
        
        elif flag == "posTrans":
            if output < -21:
                output =  -21
            elif output > 0:
                output = 0
            self.prev_output = output
            self.prev_error = error

        return output
    
    def controlePhi(self, ref, kp, ki, kd, ts, atual, flag):
        error = ref - atual        
        # print(f"erro:{error}")
        integral = error * ts
        derivative = (error - self.prev_phi_error) / ts
        #print(f"I:{integral}")
        output = (self.prev_phi_output + kp * ( error - self.prev_phi_error) + ki * integral + kd * derivative)
        output = output.item()
        
        sat = .85
        if flag == "phiAr":
            if output < -sat:
                output = -sat
            elif output > sat:
                output = sat
            self.prev_phi_output = output
            self.prev_phi_error = error
        
        elif flag == "phiAgua":
            if output < -0.75:
                output = -0.75
            elif output > 0.75:
                output = 0.75
            self.prev_phi_output = output
            self.prev_phi_error = error
    
        elif flag == "phiTrans":
            if output < -1:
                output = -1
            elif output > 1:
                output = 1
            self.prev_phi_output = output
            self.prev_phi_error = error
        
        return output
            
    def controleTheta(self, ref, kp, ki, kd, ts, atual, flag):
        error = ref - atual        
        # print(f"erro:{error}")
        integral = error * ts
        derivative = (error - self.prev_theta_error) / ts
        #print(f"I:{integral}")
        output = (self.prev_theta_output + kp * ( error - self.prev_theta_error) + ki * integral + kd * derivative)
        output = output.item()
        sat = 0.85
        if flag == "thetaAr":
            if output < -sat:
                output = -sat
            elif output > sat:
                output = sat
            self.prev_theta_output = output
            self.prev_theta_error = error
            
        elif flag == "thetaAgua":
            if output < -0.75:
                output = -0.75
            elif output > 0.75:
                output = 0.75
            self.prev_theta_output = output
            self.prev_theta_error = error
            
        elif flag == "thetaTrans":
            if output < -1:
                output = -1
            elif output > 1:
                output = 1
            self.prev_theta_output = output
            self.prev_theta_error = error
        
        return output
    
    def controlePsi(self, ref, kp, ki, kd, ts, atual, flag):
        error = ref - atual        
        # print(f"erro:{error}")
        integral = error * ts
        derivative = (error - self.prev_psi_error) / ts
        #print(f"I:{integral}")
        output = (self.prev_psi_output + kp * ( error - self.prev_psi_error) + ki * integral + kd * derivative)
        output = output.item()
        
        sat = 1
        if flag == "psiAr":
            if output < -sat:
                output = -sat
            elif output > sat:
                output = sat
            self.prev_psi_output = output
            self.prev_psi_error = error
            
        elif flag == "psiAgua":
            if output < -5:
                output = -5
            elif output > 5:
                output = 5
            self.prev_psi_output = output
            self.prev_psi_error = error
        
        elif flag == "psiTrans":
            if output < -5:
                output = -5
            elif output > 5:
                output = 5
            self.prev_psi_output = output
            self.prev_psi_error = error
        # print(output)
        
        # print(f"output: {output}")
        return output
    
    def STSMC_controllerAr(self, ts,  refz, ref_angle):

        k1z = 1.5
        k2z = 2.0
        lambdaz = .2
        k1eta = 1
        k2eta = 3
        lambda_eta = 1

        k1eta_phi = 1
        k2eta_phi = 2

        # Axi = np.array([(1/self.mass)*(s(self.angle[0])*s(self.angle[2]) + c(self.angle[1])*s(self.angle[1])*c(self.angle[2])), 
        #                 - s(self.angle[0])*c(self.angle[2]) + c(self.angle[0])*s(self.angle[1])*s(self.angle[2]),
        #                 c(self.angle[1])*c(self.angle[0])]).reshape(3, 1)
        
        # Bxi = np.array([0. , 0. , -self.gravity]).reshape(3, 1)

        # Aeta = np.array([[c(self.angle[1]), c(self.angle[1])*s(self.angle[0]), -c(self.angle[1])*c(self.angle[0])],
        #                 [0, c(self.angle[1])*c(self.angle[0]), -c(self.angle[1])*s(self.angle[0])],
        #                 [0, s(self.angle[0]), c(self.angle[0])]], dtype=float)
        
        error_z = refz - self.pos[2]
        error_z_dot = (error_z - self.prev_error)/ts

        error_phi = ref_angle[0] - self.angle[0]
        error_phi_dot = (error_phi - self.prev_phi_error )/ts

        error_theta = ref_angle[1] - self.angle[1]
        error_theta_dot = (error_theta - self.prev_theta_error)/ts

        error_psi = ref_angle[2] - self.angle[2]
        error_psi_dot = (error_psi - self.prev_psi_error)/ts
        #print(k1z*error_z_dot)
        sigmaz= k1z*error_z + k2z*error_z_dot

        # U1 = -1*(1/k2z)*np.dot(np.array(np.transpose(Axi), dtype=float),(k1z*error_z_dot + k2z*Bxi)) - lambdaz*sat(1, sigmaz)
        U1 = 3*np.sqrt(abs(sigmaz))*np.sign(sigmaz) + self.w
        wdot = .2*np.sign(sigmaz)
        self.w = self.w + wdot
        sigmaphi= k1eta_phi*error_phi+k2eta_phi*error_phi_dot
        sigmatheta= k1eta_phi*error_theta+k2eta_phi*error_theta_dot
        sigmapsi= k1eta*error_psi+k2eta*error_psi_dot

        # eta_error_dot = np.array([[error_phi_dot], [error_theta_dot], [error_psi_dot]]).reshape(3,1)
        # sat_mat = np.array([[sat(2, sigmaphi)], [sat(2, sigmatheta)], [sat(2, sigmapsi)]]).reshape(3,1)
        # Ueta = -(k1eta/k2eta)*np.dot(np.array(np.linalg.inv(Aeta), dtype=float),eta_error_dot) - lambda_eta*sat_mat
        
        Uphi = 1.5*np.sqrt(abs(sigmaphi))*np.sign(sigmaphi) + self.w1
        wdot = .08*np.sign(sigmaphi)
        self.w1 = self.w1 + wdot

        Utheta = 1*np.sqrt(abs(sigmatheta))*np.sign(sigmatheta) + self.w2
        wdot = .2*np.sign(sigmatheta)
        self.w2 = self.w2 + wdot

        Upsi = 1*np.sqrt(abs(sigmapsi))*np.sign(sigmapsi) + self.w3
        wdot = .2*np.sign(sigmapsi)
        self.w3 = self.w3 + wdot
        
        U = np.array([U1, Uphi, Utheta, Upsi]).reshape(4,1)
        self.prev_error = error_z
        self.prev_phi_error = error_phi
        self.prev_psi_error = error_psi
        self.prev_theta_error = error_theta

        return U

    def STSMC_controllerAgua(self, ts,  refz, ref_angle):

        k1z = .25
        k2z = .5
        lambdaz = .2
        k1eta = 1
        k2eta = 2
        lambda_eta = 1

        k1eta_phi = 2
        k2eta_phi = 1

        # Axi = np.array([(1/self.mass)*(s(self.angle[0])*s(self.angle[2]) + c(self.angle[1])*s(self.angle[1])*c(self.angle[2])), 
        #                 - s(self.angle[0])*c(self.angle[2]) + c(self.angle[0])*s(self.angle[1])*s(self.angle[2]),
        #                 c(self.angle[1])*c(self.angle[0])]).reshape(3, 1)
        
        # Bxi = np.array([0. , 0. , -self.gravity]).reshape(3, 1)

        # Aeta = np.array([[c(self.angle[1]), c(self.angle[1])*s(self.angle[0]), -c(self.angle[1])*c(self.angle[0])],
        #                 [0, c(self.angle[1])*c(self.angle[0]), -c(self.angle[1])*s(self.angle[0])],
        #                 [0, s(self.angle[0]), c(self.angle[0])]], dtype=float)
        
        error_z = refz - self.pos[2]
        error_z_dot = (error_z - self.prev_error)/ts

        error_phi = ref_angle[0] - self.angle[0]
        error_phi_dot = (error_phi - self.prev_phi_error )/ts

        error_theta = ref_angle[1] - self.angle[1]
        error_theta_dot = (error_theta - self.prev_theta_error)/ts

        error_psi = ref_angle[2] - self.angle[2]
        error_psi_dot = (error_psi - self.prev_psi_error)/ts
        #print(k1z*error_z_dot)
        sigmaz= k1z*error_z + k2z*error_z_dot

        # U1 = -1*(1/k2z)*np.dot(np.array(np.transpose(Axi), dtype=float),(k1z*error_z_dot + k2z*Bxi)) - lambdaz*sat(1, sigmaz)
        U1 = 20*np.sqrt(abs(sigmaz))*np.sign(sigmaz) + self.w
        wdot = 0.8*np.sign(sigmaz)
        self.w = self.w + wdot
        sigmaphi= k1eta_phi*error_phi+k2eta_phi*error_phi_dot
        sigmatheta= k1eta_phi*error_theta+k2eta_phi*error_theta_dot
        sigmapsi= (k1eta*error_psi+k2eta*error_psi_dot).round(4)

        # eta_error_dot = np.array([[error_phi_dot], [error_theta_dot], [error_psi_dot]]).reshape(3,1)
        # sat_mat = np.array([[sat(2, sigmaphi)], [sat(2, sigmatheta)], [sat(2, sigmapsi)]]).reshape(3,1)
        # Ueta = -(k1eta/k2eta)*np.dot(np.array(np.linalg.inv(Aeta), dtype=float),eta_error_dot) - lambda_eta*sat_mat
        
        Uphi = .001*np.sqrt(abs(sigmaphi))*np.sign(sigmaphi) + self.w1
        wdot = .005*np.sign(sigmaphi)
        self.w1 = self.w1 + wdot

        Utheta = .001*np.sqrt(abs(sigmatheta))*np.sign(sigmatheta) + self.w2
        wdot = .005*np.sign(sigmatheta)
        self.w2 = self.w2 + wdot

        Upsi = .01*np.sqrt(abs(sigmapsi))*np.sign(sigmapsi) + self.w3
        # print(Upsi)
        wdot = .02*np.sign(sigmapsi)
        self.w3 = self.w3 + wdot
        
        U = np.array([U1, Uphi, Utheta, Upsi]).reshape(4,1)
        self.prev_error = error_z
        self.prev_phi_error = error_phi
        self.prev_psi_error = error_psi
        self.prev_theta_error = error_theta

        return U
   
    def STSMC_controllerTransicao(self, ks, ts,  refz, ref_angle):

        k1z = .25
        k2z = .5
        lambdaz = .2
        k1eta = .46
        k2eta = 2
        lambda_eta = 1

        k1eta_phi = 2
        k2eta_phi = 1
        
        if ks == 0:
            ks = 1

        # Axi = np.array([(1/self.mass)*(s(self.angle[0])*s(self.angle[2]) + c(self.angle[1])*s(self.angle[1])*c(self.angle[2])), 
        #                 - s(self.angle[0])*c(self.angle[2]) + c(self.angle[0])*s(self.angle[1])*s(self.angle[2]),
        #                 c(self.angle[1])*c(self.angle[0])]).reshape(3, 1)
        
        # Bxi = np.array([0. , 0. , -self.gravity]).reshape(3, 1)

        # Aeta = np.array([[c(self.angle[1]), c(self.angle[1])*s(self.angle[0]), -c(self.angle[1])*c(self.angle[0])],
        #                 [0, c(self.angle[1])*c(self.angle[0]), -c(self.angle[1])*s(self.angle[0])],
        #                 [0, s(self.angle[0]), c(self.angle[0])]], dtype=float)
        
        error_z = refz - self.pos[2]
        error_z_dot = (error_z - self.prev_error)/ts

        error_phi = ref_angle[0] - self.angle[0]
        error_phi_dot = (error_phi - self.prev_phi_error )/ts

        error_theta = ref_angle[1] - self.angle[1]
        error_theta_dot = (error_theta - self.prev_theta_error)/ts

        error_psi = ref_angle[2] - self.angle[2]
        error_psi_dot = (error_psi - self.prev_psi_error)/ts
        #print(k1z*error_z_dot)
        sigmaz= k1z * error_z  + k2z * error_z_dot 

        # U1 = -1*(1/k2z)*np.dot(np.array(np.transpose(Axi), dtype=float),(k1z*error_z_dot + k2z*Bxi)) - lambdaz*sat(1, sigmaz)
        U1 = 20 / ks * np.sqrt(abs(sigmaz)) * np.sign(sigmaz) + self.w
        wdot = 0.8 * np.sign(sigmaz) / ks
        self.w = self.w + wdot
        sigmaphi= k1eta_phi  * error_phi * 2 + k2eta_phi * error_phi_dot
        sigmatheta= k1eta_phi * error_theta + k2eta_phi * error_theta_dot
        sigmapsi= (k1eta * error_psi  + k2eta * error_psi_dot).round(4)

        # eta_error_dot = np.array([[error_phi_dot], [error_theta_dot], [error_psi_dot]]).reshape(3,1)
        # sat_mat = np.array([[sat(2, sigmaphi)], [sat(2, sigmatheta)], [sat(2, sigmapsi)]]).reshape(3,1)
        # Ueta = -(k1eta/k2eta)*np.dot(np.array(np.linalg.inv(Aeta), dtype=float),eta_error_dot) - lambda_eta*sat_mat
        
        Uphi = .001 * np.sqrt(abs(sigmaphi)) * np.sign(sigmaphi) + self.w1
        wdot = .005 * np.sign(sigmaphi)
        self.w1 = self.w1 + wdot

        Utheta = .001 * np.sqrt(abs(sigmatheta)) * np.sign(sigmatheta) + self.w2
        wdot = .005 * np.sign(sigmatheta)
        self.w2 = self.w2 + wdot

        Upsi = .2 * np.sqrt(abs(sigmapsi)) * np.sign(sigmapsi) + self.w3
        # print(Upsi)
        wdot = .1 * np.sign(sigmapsi)
        self.w3 = self.w3 + wdot
        
        U = np.array([U1, Uphi, Utheta, Upsi]).reshape(4,1)
        self.prev_error = error_z
        self.prev_phi_error = error_phi
        self.prev_psi_error = error_psi
        self.prev_theta_error = error_theta

        return U

class QuadcopterAr(Quadcopter):

    def __init__(self, pos, vel, angle, ang_vel, dt):
        super().__init__(pos, vel, angle, ang_vel, dt)

        # Ar
        self.rho = 1.29  # Densidade do ar (kg/m^3)
        self.kQ = 1.126e-4  # Coeficiente de torque da hélice no ar
        self.cT = 3.5e-2  # Coeficiente de sustentação da hélice no ar
        # constante de proporcionalidade para converter a velocidade rotacional do motor em empuxo (T = kt * omega^2), N/(rpm)^2
        self.kT = self.cT * self.rho * self.r**4 * np.pi
        self.Jr = 8.61e-4  # Momento de inércia do motor e hélice (kg/m^2)
        # constante de proporcionalidade para converter a velocidade do motor em torque (torque = b*omega^2), (N*m)/(rpm)^2
        self.b_prop = self.kQ
        self.thrust = self.kT * \
            (self.speeds[0]**2 + self.speeds[1]**2 +
             self.speeds[2]**2 + self.speeds[3]**2)

        # Potencia maxima dos motores
        self.rpm_max = 8500  # Velocidade máxima de rotação em rpm na água 8500
        self.P_max = 202.8  # Potência máxima em Watts
        self.N_max = 2 * np.pi * self.rpm_max / 60  # Conversão de rpm para rad/s
        self.minT = 0  # Empuxo mínimo de qualquer motor individual, em N
        self.r = 0.15  # Raio da hélice (m)
        self.maxT = self.N_max
        print(f"Maximo motores Ar: {self.maxT}")

        # Matriz para conversão do controle para speeds
        self.K = np.array([
            [1, 1, 1, 1],
            [self.L, -self.L, -self.L, self.L],
            [self.L, self.L, -self.L, -self.L],
            [self.kQ / self.kT, -self.kQ / self.kT, self.kQ / self.kT, -self.kQ / self.kT]
        ], dtype=float)# * self.kT
        
        
        self.K_inv = np.array(np.linalg.inv(self.K), dtype=float)
        
        

    def ganhosPosAr(self):

        kp = 1000
        ki = 2000
        kd = 15

        return kp, ki, kd
    
    def ganhosPhiAr(self):

        kp = 0.6
        ki = 43.5
        kd = 10
        
        return kp, ki, kd
    
    def ganhosThetaAr(self):

        kp = 0.6
        ki = 43.5
        kd = 10
        
        return kp, ki, kd
    
    def ganhosPsiAr(self):

        kp = 0.02
        ki = 1.25
        kd = 10
        
        return kp, ki, kd

    def getAccelAr(self, angle, ang_vel,  speeds):
        
        """
        Calcula as acelerações lineares e angulares do quadrotor no ar.

        Returns:
        - tuple: Acelerações lineares e angulares calculadas.
        """
        self.angle = angle
        self.ang_vel = ang_vel
        self.speeds = speeds
        # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}")

        U1 = self.kT * (self.speeds[0]**2 + self.speeds[1]
                        ** 2 + self.speeds[2]**2 + self.speeds[3]**2)
        U2 = (np.sqrt(2) * self.kT * self.L) / 2 * \
            (self.speeds[0]**2 - self.speeds[1]**2 -
             self.speeds[2]**2 + self.speeds[3]**2)
        U3 = (np.sqrt(2) * self.kT * self.L) / 2 * \
            (self.speeds[0]**2 + self.speeds[1]**2 -
             self.speeds[2]**2 - self.speeds[3]**2)
        U4 = self.b_prop * \
            (self.speeds[0]**2 - self.speeds[1]**2 +
             self.speeds[2]**2 - self.speeds[3]**2)
        # print(f"U1: {U1}")
        # Força total (U1)
        accel_x = U1 / self.mass * (c(self.angle[2]) * s(self.angle[1]) * c(
            self.angle[0]) + s(self.angle[2]) * s(self.angle[0]))
        accel_y = U1 / self.mass * (s(self.angle[2]) * s(self.angle[1]) * c(
            self.angle[0]) - c(self.angle[2]) * s(self.angle[0]))
        accel_z = U1 / self.mass * \
            c(self.angle[1]) * c(self.angle[0]) - self.gravity

        # Momento de Rotação (Roll) (U2)
        accel_phi = np.sqrt(2)* self.L / 2 * U2 
        # - self.ang_vel[1] * self.Jr * (self.speeds[0] - self.speeds[1] + self.speeds[2] - self.speeds[3])

        # Momento de Inclinação (Pitch) (U3)
        accel_theta = np.sqrt(2)* self.L / 2 * U3 
        # + self.ang_vel[0] * self.Jr * (self.speeds[0] - self.speeds[1] + self.speeds[2] - self.speeds[3])

        # Momento de Guinada (Yaw) (U4)
        accel_psi = (self.b_prop / self.kT) * U4
        
        # print(f"dentro: {self.speeds}")
        self.lin_acc = np.array(
            [accel_x, accel_y, accel_z], dtype=float
        ).round(4).reshape(3, 1)

        self.ang_acc = np.array(
            [accel_phi, accel_theta, accel_psi], dtype=float
        ).round(4).reshape(3, 1)

        
        # print(f"Acelerações no ar: {self.lin_acc}, {self.ang_acc}")
        return self.lin_acc, self.ang_acc

class QuadcopterAgua(Quadcopter):

    def __init__(self, pos, vel, angle, ang_vel, dt):
        super().__init__(pos, vel, angle, ang_vel, dt)

        # Agua
        self.pos = pos
        self.vel = vel
        self.angle = angle
        self.ang_vel = ang_vel
        # np.array([0, 0, -self.gravity], dtype = float).reshape(3,1)
        self.g = self.gravity
        self.rho = 1000  # Densidade da água (kg/m^3)
        self.buoyancy_center = (0, 0, -0.02)  # Coordenadas do centro de empuxo
        self.Cdw = 0.9  # Coeficiente de resistência adimensional na água
        self.kx = 0.8  # Coeficiente de resistência à rotação em torno do eixo X na água
        self.ky = 1  # Coeficiente de resistência à rotação em torno do eixo Y na água
        self.kz = 0.8  # Coeficiente de resistência à rotação em torno do eixo Z na água
        self.cT = 4.97e-6  # Coeficiente de sustentação da hélice na água
        # constante de proporcionalidade para converter a velocidade rotacional do motor em empuxo (T = kt * self.speeds^2), N/(rpm)^2
        self.kT = self.cT * self.rho * self.r**4 * np.pi
        self.kQ = 2.012e-5  # Coeficiente de torque da hélice na água
        self.b_prop = self.kQ
        self.Jr = 8.61e-4  # Momento de inércia do motor e hélice (kg/m^2)
        self.Sx = 1.05e-2  # Área na direção X (m^2)
        self.Sy = 1.96e-2  # Área na direção Y (m^2)
        self.Sz = 4.2e-2  # Área na direção Z (m^2)
        self.S = np.diag([self.Sx, self.Sy, self.Sz])
        self.Jxx = 2.365e-2  # Inércia de rotação em torno do eixo X (kg/m^2)
        self.Jyy = 1.318e-2  # Inércia de rotação em torno do eixo Y (kg/m^2)
        self.Jzz = 1.318e-2  # Inércia de rotação em torno do eixo Z (kg/m^2)

        ma = (4/6) * np.pi * self.rho * self.r_esfera**3
        Ia = (1/5) * ma * self.r_esfera**2
        self.Mt = self.mass + 3 * ma + 3 * Ia
        # self.Mt = self.Mt[:3, :3] #  parte superior esquerda 3x3
        # print(self.Mt)
        # Redimensionar a matriz self.Mt para um escala
        # self.Mt = np.trace(self.Mt ) / 6
        # print(self.Mt)

        # Potencia maxima
        self.rpm_max = 8500  # Insira a velocidade máxima de rotação em rpm no ar
        self.P_max = 202.8  # Potência máxima em Watts
        self.N_max = 2 * np.pi * self.rpm_max / 60  # Conversão de rpm para rad/s
        self.minT = 0 # empuxo mínimo de qualquer motor individual, em N
        self.r = 0.15  # Raio da hélice (m)
        self.maxT = self.N_max

        # self.thrust = self.kT * \
        #     (self.speeds[0]**2 + self.speeds[1]**2 +
        #      self.speeds[2]**2 + self.speeds[3]**2)
        # Matriz para conversão do controle para speeds
        self.K = np.array([
            [1, 1, 1, 1],
            [self.L, -self.L, -self.L, self.L],
            [self.L, self.L, -self.L, -self.L],
            [self.kQ / self.kT, -self.kQ / self.kT,
                self.kQ / self.kT, -self.kQ / self.kT]
        ])
        # self.K_inv = - np.array(np.linalg.inv(self.K), dtype=float) # observe o menos para inverter
        # self.K  = np.dot(-1, self.K)
        self.K_inv = - np.array(np.linalg.inv(self.K), dtype=float) # observe o menos para inverter
        print(f"antes:{self.K_inv}")
        self.K_inv  = np.dot(-1, self.K_inv)
        print(f"depois:{np.dot(self.K_inv, np.ones(4))}")
    def ganhosPosAgua(self):

        # posição
        kp = 9000
        ki = 50000
        kd = 10

        return kp, ki, kd
    
    def ganhosPhiAgua(self):

        kp = 0.006
        ki = 0.0017
        kd = 0.0053
        
        return kp, ki, kd
    
    def ganhosThetaAgua(self):

        kp = 0.006
        ki = 0.0017
        kd = 0.0053
        
        return kp, ki, kd
    
    def ganhosPsiAgua(self):

        kp = 10
        ki = 0.09
        kd = 0.007
        
        
        return kp, ki, kd

    def calculate_water_buoyancy_and_drag(self, angle, ang_vel, vel, speeds):
        
        """
        Calcula as acelerações lineares e angulares do quadrotor no ar.

        Returns:
        - tuple: Acelerações lineares e angulares calculadas.
        """
        self.angle = angle
        self.ang_vel = ang_vel
        self.vel = vel
        self.speeds = speeds
        
        # Cálculo da força de flutuabilidade
        Fb_b = np.array([
            -self.rho * self.gravity * self.V * np.sin(self.angle[1]),
            self.rho * self.gravity * self.V *
            np.sin(self.angle[0]) * np.cos(self.angle[1]),
            self.rho * self.gravity * self.V *
            np.cos(self.angle[1]) * np.cos(self.angle[0])
        ], dtype=float)

        # Cálculo do momento de empuxo e torque de empuxo
        Mb_b = -Fb_b * np.array([
            self.buoyancy_center[2] * np.sin(self.angle[0]) * np.cos(
                self.angle[1]) - self.buoyancy_center[1] * np.cos(self.angle[0]) * np.cos(self.angle[1]),
            self.buoyancy_center[0] * np.cos(self.angle[0]) * np.cos(
                self.angle[1]) + self.buoyancy_center[2] * np.sin(self.angle[1]),
            -self.buoyancy_center[1] * np.sin(self.angle[0]) - self.buoyancy_center[0] * np.cos(
                self.angle[0]) * np.cos(self.angle[1])
        ], dtype=float)

        Mbxw = (-np.sqrt(2) / 2 * self.kT * self.L * (self.speeds[0]**2 - self.speeds[1]**2 - self.speeds[2]**2 + self.speeds[3]**2) -
                self.kx * np.abs(self.ang_vel[0]) * self.ang_vel[0] -
                np.dot(Fb_b , (self.buoyancy_center[2] * np.cos(self.angle[1]) * np.sin(self.angle[0]) -
                 self.buoyancy_center[1] * np.cos(self.angle[1]) * np.cos(self.angle[0]))) +
                self.Jr * self.ang_vel[1] * (self.speeds[0] - self.speeds[1]
                                             + self.speeds[2]- self.speeds[3])
                ).round(4)
        # print(f"as tripas: {self.ky}, {np.abs(self.ang_vel[1])}, {self.ang_vel[1]} ")
        # print(f"parte1: {-np.sqrt(2) / 2 * self.kT * self.L * (self.speeds[0]**2 + self.speeds[1]**2 - self.speeds[2]**2 - self.speeds[3]**2)}, parte2: {self.ky * np.abs(self.ang_vel[1]) * self.ang_vel[1]}, parte3:{Fb_b *(self.buoyancy_center[0] * np.cos(self.angle[1]) * np.cos(self.angle[0]) - self.buoyancy_center[2] * np.sin(self.angle[1]))}, parte4: {self.Jr * self.ang_vel[0] * (self.speeds[0]**2 - self.speeds[1]** 2 + self.speeds[2]**2 - self.speeds[3]**2)}")
        Mbyw = (-np.sqrt(2) / 2 * self.kT * self.L * (self.speeds[0]**2 + self.speeds[1]**2 - self.speeds[2]**2 - self.speeds[3]**2) -
                self.ky * np.abs(self.ang_vel[1]) * self.ang_vel[1] -
                np.dot(Fb_b, (self.buoyancy_center[0] * np.cos(self.angle[1]) * np.cos(self.angle[0]) -
                 self.buoyancy_center[2] * np.sin(self.angle[1]))) -
                self.Jr * self.ang_vel[0] * (self.speeds[0] - self.speeds[1]
                                             + self.speeds[2] - self.speeds[3])
                ).round(4)

        Mbzw = (-self.kQ * (self.speeds[0]**2 - self.speeds[1]**2 + self.speeds[2]**2 - self.speeds[3]**2) -
                self.kz * np.abs(self.ang_vel[2]) * self.ang_vel[2] -
                np.dot(Fb_b,
                (-self.buoyancy_center[1] * np.sin(self.angle[1]) -
                 self.buoyancy_center[0] * np.cos(self.angle[1]) * np.cos(self.angle[0]))
                )).round(4)
        # print(f"Matrizes agua: {Mbxw}, {Mbyw}, {Mbzw}")
        return Fb_b, Mb_b, Mbxw, Mbyw, Mbzw

    def getAccelAgua(self, angle, ang_vel, vel, speeds):
        '''Calcula as dinâmicas translacionais e rotacionais do sistema quadrotor submerso na água.       '''
  
        self.speeds = speeds
        self.angle = angle
        self.ang_vel = ang_vel
        self.vel = vel
        _ , self.thrust, self.Mbxw, self.Mbyw, self.Mbzw = self.calculate_water_buoyancy_and_drag(self.angle, self.ang_vel, self.vel, self.speeds)

        accel_x = np.float64(((1 / self.Mt) * ((self.Mt * self.g - self.rho * self.g * self.V) * np.sin(self.angle[1]) +
                                    0.5 * self.rho * self.Cdw * self.S[0, 0] * np.abs(self.vel[0]) * self.vel[0]) -
                   self.vel[2] * self.ang_vel[1] +
                   self.vel[1] * self.ang_vel[2]
                   ))

        accel_y = np.float64(((1 / self.Mt) * (-(self.Mt * self.g - self.rho * self.g * self.V) * np.sin(self.angle[0]) * np.cos(self.angle[1])+
                                    0.5 * self.rho * self.Cdw * self.S[1, 1] * np.abs(self.vel[1]) * self.vel[1]) -
                   self.vel[0] * self.ang_vel[2] +
                   self.vel[2] * self.ang_vel[0]
                   ))
        # print(f"Tentando: {-(self.Mt * self.g - self.rho * self.g * self.V)}, parte1: {-(self.Mt * self.g - self.rho * self.g * self.V) * np.cos(self.angle[1]) * np.cos(self.angle[0])}, parte2: {-0.5 * self.rho * self.Cdw * self.S[2, 2] * np.abs(self.vel[2]) * self.vel[2]}, Parte3: {-(self.kT * (self.speeds[0]**2 + self.speeds[1]**2 + self.speeds[2]**2 + self.speeds[3]**2))}")
    
        accel_z = np.float64((1 / self.Mt) * (-(self.Mt * self.g - self.rho * self.g * self.V) * np.cos(self.angle[1]) * np.cos(self.angle[0]) -
                                   0.5 * self.rho * self.Cdw * self.S[2, 2] * np.abs(self.vel[2]) * self.vel[2] -
                                   (self.kT *
                                   (self.speeds[0]**2 + self.speeds[1]**2 +
                                    self.speeds[2]**2 + self.speeds[3]**2))
                                   ))

        accel_phi = np.float64((1 / self.Jxx) * (self.Mbxw[0] + self.Mbxw[1] + self.Mbxw[2]) + (
            (self.Jyy - self.Jzz) / self.Jxx) * self.ang_vel[1] * self.ang_vel[2])

        accel_theta = np.float64((1 / self.Jyy) * (self.Mbyw[0] + self.Mbyw[1] + self.Mbyw[2]) + (
            (self.Jzz - self.Jxx) / self.Jxx) * self.ang_vel[2] * self.ang_vel[0])

        accel_psi = np.float64((1 / self.Jzz) * (self.Mbzw[0] + self.Mbzw[1] + self.Mbzw[2]) + (
            (self.Jxx - self.Jyy) / self.Jzz) * self.ang_vel[0] * self.ang_vel[1])
        
        # print(f"Acelerações lineares => x:{accel_x},teste:{0.5 * self.rho * self.Cdw * self.S[2, 2] * np.abs(self.vel[2]) * self.vel[2]},z:{accel_z}")

        self.lin_acc = np.array(
            [accel_x, accel_y, accel_z], dtype=float
        ).round(4).reshape(3, 1)

        self.ang_acc = np.array(
            [accel_phi, accel_theta, accel_psi], dtype=float
        ).round(4).reshape(3, 1)

        # print(f"Acelerações na agua: {self.lin_acc}, {self.ang_acc}")
        
        return self.lin_acc, self.ang_acc

class QuadcopterTransicao(QuadcopterAgua):

    def __init__(self, pos, vel, angle, ang_vel, dt):
        super().__init__(pos, vel, angle, ang_vel, dt)
        self.pos = pos
        self.vel = vel
        self.angle = angle
        self.ang_vel = ang_vel
        
        self.K = np.array([
            [1, 1, 1, 1],
            [self.L, -self.L, -self.L, self.L],
            [self.L, self.L, -self.L, -self.L],
            [self.kQ / self.kT, -self.kQ / self.kT,
                self.kQ / self.kT, -self.kQ / self.kT]
        ])
        # self.K_inv = - np.array(np.linalg.inv(self.K), dtype=float) # observe o menos para inverter
        # self.K  = np.dot(-1, self.K)
        self.K_inv = - np.array(np.linalg.inv(self.K), dtype=float) # observe o menos para inverter
        self.K_inv  = np.dot(-1, self.K_inv)

    def ganhosPosTransicao(self):

        kp = 90000
        ki = 500000
        kd = 10

        return kp, ki, kd
    
    def ganhosPhiTrans(self):

        kp = 0.006
        ki = 0.0017
        kd = 0.0053
        
        return kp, ki, kd
    
    def ganhosThetaTrans(self):

        kp = 0.006
        ki = 0.0017
        kd = 0.0053
        
        return kp, ki, kd
    
    def ganhosPsiTrans(self):

        kp = 1
        ki = 0.09
        kd = 0.007
        
        return kp, ki, kd

    def getAccelTrans(self, pos, angle, ang_vel, vel, speeds):
        '''Calcula as dinâmicas translacionais e rotacionais do sistema quadrotor submerso na água.       '''
       
        self.pos = pos
        self.angle = angle
        self.ang_vel = ang_vel
        self.vel = vel
        self.speeds = speeds
        
        # Coeficiente crítico
        epsilon = 0.25  # faixa de transição
        self.Ks = (0.5 * (1 - self.pos[2] / epsilon)).round(4)
        
        _ , self.thrust, self.Mbxw, self.Mbyw, self.Mbzw = super().calculate_water_buoyancy_and_drag(self.angle, self.ang_vel, self.vel, self.speeds)


        accel_x = np.float64(((1 / self.Mt) * ((self.Mt * self.g - self.rho * self.g * self.V) * np.sin(self.angle[1]) +
                                    0.5 * self.Ks * self.rho * self.Cdw * self.S[0, 0] * np.abs(self.vel[0]) * self.vel[0]) -
                   self.vel[2] * self.ang_vel[1] +
                   self.vel[1] * self.ang_vel[2]
                   ))

        accel_y = np.float64(((1 / self.Mt) * (-(self.Mt * self.g - self.rho * self.g * self.V) * np.sin(self.angle[0]) * np.cos(self.angle[1])+
                                    0.5 * self.Ks * self.rho * self.Cdw * self.S[1, 1] * np.abs(self.vel[1]) * self.vel[1]) -
                   self.vel[0] * self.ang_vel[2] +
                   self.vel[2] * self.ang_vel[0]
                   ))
        # print(f"Tentando: {-(self.Mt * self.g - self.rho * self.g * self.V)}, parte1: {-(self.Mt * self.g - self.rho * self.g * self.V) * np.cos(self.angle[1]) * np.cos(self.angle[0])}, parte2: {-0.5 * self.rho * self.Cdw * self.S[2, 2] * np.abs(self.vel[2]) * self.vel[2]}, Parte3: {-(self.kT * (self.speeds[0]**2 + self.speeds[1]**2 + self.speeds[2]**2 + self.speeds[3]**2))}")
    
        accel_z = np.float64((1 / self.Mt) * (-(self.Mt * self.g - self.rho * self.g * self.V) * np.cos(self.angle[1]) * np.cos(self.angle[0]) -
                                   0.5 * self.Ks * self.rho * self.Cdw * self.S[2, 2] * np.abs(self.vel[2]) * self.vel[2] -
                                   (self.kT *
                                   (self.speeds[0]**2 + self.speeds[1]**2 +
                                    self.speeds[2]**2 + self.speeds[3]**2))
                                   ))

        accel_phi = np.float64((1 / self.Jxx) * (self.Mbxw[0] + self.Mbxw[1] + self.Mbxw[2]) + (
            (self.Jyy - self.Jzz) / self.Jxx) * self.ang_vel[1] * self.ang_vel[2])

        accel_theta = np.float64((1 / self.Jyy) * self.Ks * (self.Mbyw[0] + self.Mbyw[1] + self.Mbyw[2]) + (
            (self.Jzz - self.Jxx) / self.Jxx) * self.ang_vel[2] * self.ang_vel[0])

        accel_psi = np.float64((1 / self.Jzz) * self.Ks * (self.Mbzw[0] + self.Mbzw[1] + self.Mbzw[2]) + (
            (self.Jxx - self.Jyy) / self.Jzz) * self.ang_vel[0] * self.ang_vel[1])

        # print(f"Acelerações lineares => x:{accel_x},teste:{0.5 * self.rho * self.Cdw * self.S[2, 2] * np.abs(self.vel[2]) * self.vel[2]},z:{accel_z}")

        self.lin_acc = np.array(
            [accel_x, accel_y, accel_z], dtype=float
        ).round(4).reshape(3, 1)

        self.ang_acc = np.array(
            [accel_phi, accel_theta, accel_psi], dtype=float
        ).round(4).reshape(3, 1)
        # print(f"Acelerações na transição: {self.lin_acc}, {self.ang_acc}")
        
        return self.lin_acc, self.ang_acc
class QuadcopterSim(Quadcopter):

    def __init__(self, ref, pos, vel, angle, ang_vel, dt):
        super().__init__(pos, vel, angle, ang_vel, dt)

        # print(f"pos:{self.pos[2]}")
        # atribuições
        self.pos = pos
        self.vel = vel
        self.angle = angle
        self.ang_vel = ang_vel
        self.sAtitude = np.array([0., 0., 0.], dtype = float).reshape(3,1)
        self.erro_atitude = np.array([self.prev_phi_error, self.prev_theta_error, self.prev_psi_error], dtype = float).reshape(3,1)
        self.ref = ref
        self.w = 0
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        
        self.quadAr = QuadcopterAr(self.pos, self.vel,
                              self.angle, self.ang_vel, self.dt)
        self.quadAgua = QuadcopterAgua(
            self.pos, self.vel, self.angle, self.ang_vel, self.dt)
        self.quadTrans = QuadcopterTransicao(
            self.pos, self.vel, self.angle, self.ang_vel, self.dt)
        
        self.meio = 0

    def stepSTSM(self):
        # print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")

        # Determina o ambiente (Ar, Água, ou Transição) com base na altura (pos[2])
        # AR
        if self.pos[2] > 0.25:
            
            if self.meio != 1:
                self.w = 0
                self.w1 = 0
                self.w2 = 0
                self.w3 = 0
            
            self.meio = 1
            # print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            # print(f"To no Ar {self.pos[2]}")
            self.kT = self.quadAr.kT
            self.K_inv = self.quadAr.K_inv
            self.maxT = self.quadAr.maxT
            self.minT = self.quadAr.minT
            
            U = np.array(self.STSMC_controllerAr(self.dt, self.ref[2], np.array([[0], [0], [0]])), dtype=float).reshape(4,1)
            # print(f"1:{U}")
            Usat = np.array(self.satcontrol(U,"ar"), dtype=float)
            # print(f"2:{U}")
            self.speeds = self.des2speeds(Usat)
            # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}")
            self.sAltitude = np.array([Usat[0]], dtype=float)
            self.erro_altitude = np.array([self.prev_error], dtype=float)
            
            self.sAtitude =  np.array([Usat[1], Usat[2], Usat[3]], dtype=float)
            print()
            self.erro_atitude = np.array([self.prev_phi_error, self.prev_theta_error, self.prev_psi_error], dtype = float).reshape(3,1)
            self.lin_acc, self.ang_acc = self.quadAr.getAccelAr(self.angle, self.ang_vel, self.speeds)
            
            
        # Agua
        elif self.pos[2] < -0.25:
            
            
            if self.meio != 2:
                self.w = 0
                self.w1 = 0
                self.w2 = 0
                self.w3 = 0
            
            self.meio = 2
            # print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            # print(f"To na agua {self.pos[2]}")
            self.kT = self.quadAgua.kT
            # print(self.kT)
            self.K_inv = self.quadAgua.K_inv
            self.maxT = self.quadAgua.maxT
            self.minT = self.quadAgua.minT

            U = np.array(self.STSMC_controllerAgua(self.dt, self.ref[2], np.array([[0], [0], [0]])), dtype=float).reshape(4,1)
            # print(f"1:{U}")
            Usat = np.array(self.satcontrol(U,"agua"), dtype=float)
            # print(f"2:{U}")
            self.speeds = self.des2speeds(-Usat)
            # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}")
            self.sAltitude = np.array([Usat[0]], dtype=float)
            self.erro_altitude = np.array([self.prev_error], dtype=float)
            
            self.sAtitude =  np.array([Usat[1], Usat[2], Usat[3]], dtype=float)
            self.erro_atitude = np.array([self.prev_phi_error, self.prev_theta_error, self.prev_psi_error], dtype = float).reshape(3,1)
            self.lin_acc, self.ang_acc = self.quadAgua.getAccelAgua(self.angle, self.ang_vel, self.vel, self.speeds)
            

        # Transição
        # elif -0.25 <= self.pos[2] <= 0.25:
        elif (-0.25 <= self.pos[2] and self.pos[2] <= 0.25):
            
            if self.meio != 3:
                self.speeds = np.zeros(4)
                self.w = 0
                self.w1 = 0
                self.w2 = 0
                self.w3 = 0
                
            self.meio = 3
            
            # Coeficiente crítico
            epsilon = 0.25  # faixa de transição
            Ks_value = (0.5 * (1 - self.pos[2] / epsilon)).round(4)
            # self.vel = np.array([0., 0., 0.], dtype = float).reshape(3,1)
            # print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            print(f"To na transição {self.pos[2]}")
            # print(f"Ks:{self.Ks}")
            
            self.kT = self.quadAgua.kT
            self.K_inv = self.quadAgua.K_inv
            self.maxT = self.quadAgua.maxT
            self.minT = self.quadAgua.minT
                       
            U = np.array(self.STSMC_controllerTransicao(Ks_value, self.dt, self.ref[2], np.array([[0], [0], [0]])), dtype=float).reshape(4,1)
            # print(f"1:{U}")
            
            Usat = np.array(self.satcontrol(U,"transicao"), dtype=float)
            # print(f"2:{U}")
            self.speeds = self.des2speeds(-Usat)
            self.sAltitude = np.array([Usat[0]], dtype=float)
            self.erro_altitude = np.array([self.prev_error], dtype=float)
            
            self.sAtitude =  np.array([Usat[1], Usat[2], Usat[3]], dtype=float)
            self.erro_atitude = np.array([self.prev_phi_error, self.prev_theta_error, self.prev_psi_error], dtype = float).reshape(3,1)
            # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}")
            self.lin_acc, self.ang_acc = self.quadTrans.getAccelTrans(self.pos, self.angle, self.ang_vel, self.vel, self.speeds)

        else:
            
            print("Erro no step")
            print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            
            print(f"Shape das variáveis: pos={np.array(self.pos).shape}, vel={self.vel.shape}, angle={self.angle.shape}, ang_vel={self.ang_vel.shape}, lin_acc={self.lin_acc.shape}, ang_acc={self.ang_acc.shape}")

        # Ajustando os shapes
        # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}, posição: {self.pos}")
        self.pos = np.round(self.pos.reshape(3, 1), 4)
        self.vel = np.round(self.vel.reshape(3, 1), 4)
        self.angle = np.round(self.angle.reshape(3, 1), 4)
        self.ang_vel = np.round(self.ang_vel.reshape(3, 1), 4)
        self.lin_acc = np.round(self.lin_acc.reshape(3, 1), 4)
        self.ang_acc = np.round(self.ang_acc.reshape(3, 1), 4)

        self.vel += self.dt * self.lin_acc
        self.ang_vel += self.dt * self.ang_acc

        # self.find_vel_lin_body()
        # self.find_vel_ang_body()

        self.angle += self.dt * self.ang_vel
        self.angle = self.sat_angle()
        self.pos += self.dt * self.vel
        # if self.pos[2] < 0:
        #     self.pos[2] = 0
        #     self.vel = np.array([0., 0., 0.], dtype = float).reshape(3,1)
        self.time += self.dt


    def stepPID(self):
        
        #print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")

        # Determina o ambiente (Ar, Água, ou Transição) com base na altura (pos[2])
        # AR
        if self.pos[2] >= 0.12:
            
            # print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            # print(f"To no Ar {self.pos[2]}")
            self.kT = self.quadAr.kT
            self.K_inv = self.quadAr.K_inv
            self.maxT = self.quadAr.maxT
            self.minT = self.quadAr.minT
            
            # Controle de posição
            self.Kp, self.Ki, self.Kd = self.quadAr.ganhosPosAr()
            thrust_needed = self.controlePos(self.ref[2], self.Kp, self.Ki, self.Kd, self.dt, self.pos[2], 'posAr')
            # thrust_needed = 0 #self.mass * self.gravity
            
            # Controle de Phi
            self.KpPhi, self.KiPhi, self.KdPhi = self.quadAr.ganhosPhiAr()
            phi_needed = self.controlePhi(0, self.KpPhi, self.KiPhi, self.KdPhi, self.dt, self.angle[0], 'phiAr')
            
            # Controle de Theta
            self.KpTheta, self.KiTheta, self.KdTheta = self.quadAr.ganhosThetaAr()
            theta_needed = self.controleTheta(0, self.KpTheta, self.KiTheta, self.KdTheta, self.dt, self.angle[1], 'thetaAr')
            
            # Controle de Psi
            self.KpPsi, self.KiPsi, self.KdPsi = self.quadAr.ganhosPsiAr()
            psi_needed = self.controlePsi(0, self.KpPsi, self.KiPsi, self.KdPsi, self.dt, self.angle[2], 'psiAr')
        
            
            self.sAltitude = np.array([thrust_needed], dtype=float)
            self.erro_altitude = np.array([self.prev_error], dtype=float)
            
            self.sAtitude =  np.array([phi_needed, theta_needed, psi_needed], dtype=float)
            self.erro_atitude = np.array([self.prev_phi_error, self.prev_theta_error, self.prev_psi_error], dtype = float).reshape(3,1)
            
            # U = np.array([thrust_needed + self.mass * self.gravity, phi_needed, theta_needed, psi_needed], dtype=float)
            U = np.array([thrust_needed, phi_needed, theta_needed, psi_needed], dtype=float)
            self.speeds = self.des2speeds(U)
            # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}")
            self.lin_acc, self.ang_acc = self.quadAr.getAccelAr(self.angle, self.ang_vel, self.speeds)
            
            
        # Agua
        elif self.pos[2] <= -0.25:
           
            # print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            print(f"To na agua {self.pos[2]}")
            self.kT = self.quadAgua.kT
            # print(self.kT)
            self.K_inv = self.quadAgua.K_inv
            self.maxT = self.quadAgua.maxT
            self.minT = self.quadAgua.minT
            
            # Controle de posição
            self.Kp, self.Ki, self.Kd = self.quadAgua.ganhosPosAgua()
            thrust_needed = self.controlePos(self.ref[2], self.Kp, self.Ki, self.Kd, self.dt, self.pos[2], 'posAgua')
            # thrust_needed = 0 #self.mass * self.gravity
            
            # Controle de Phi
            self.KpPhi, self.KiPhi, self.KdPhi = self.quadAgua.ganhosPhiAgua()
            phi_needed = self.controlePhi(0, self.KpPhi, self.KiPhi, self.KdPhi, self.dt, self.angle[0], 'phiAgua')
            
            # Controle de Theta
            self.KpTheta, self.KiTheta, self.KdTheta = self.quadAgua.ganhosThetaAgua()
            theta_needed = self.controleTheta(0, self.KpTheta, self.KiTheta, self.KdTheta, self.dt, self.angle[1], 'thetaAgua')
            
            # Controle de Psi
            self.KpPsi, self.KiPsi, self.KdPsi = self.quadAgua.ganhosPsiAgua()
            psi_needed = self.controlePsi(0, self.KpPsi, self.KiPsi, self.KdPsi, self.dt, self.angle[2], 'psiAgua')
            
            
            self.sAltitude = np.array([thrust_needed], dtype=float)
            self.erro_altitude = np.array([self.prev_error], dtype=float)
            
            self.sAtitude =  np.array([phi_needed, theta_needed, psi_needed], dtype=float)
            self.erro_atitude = np.array([self.prev_phi_error, self.prev_theta_error, self.prev_psi_error], dtype = float).reshape(3,1)
            
            U = np.array([thrust_needed, phi_needed, theta_needed, psi_needed], dtype=float)
            
            self.speeds = self.des2speeds(-U)
            # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}")
            self.lin_acc, self.ang_acc = self.quadAgua.getAccelAgua(self.angle, self.ang_vel, self.vel, self.speeds)
            

        # Transição
        # elif -0.25 <= self.pos[2] <= 0.25:
        elif (-0.25 < self.pos[2] and self.pos[2] < 0.25):
            
            # self.vel = np.array([0., 0., 0.], dtype = float).reshape(3,1)
            # print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            print(f"To na transição {self.pos[2]}")
            # print(f"Ks:{self.Ks}")
            
            self.kT = self.quadAgua.kT
            self.K_inv = self.quadAgua.K_inv
            self.maxT = self.quadAgua.maxT
            self.minT = self.quadAgua.minT
            
            # Controle de posição
            self.Kp, self.Ki, self.Kd = self.quadTrans.ganhosPosTransicao()
            thrust_needed = self.controlePos(self.ref[2], self.Kp, self.Ki, self.Kd, self.dt, self.pos[2], 'posTrans')
            
            # Controle de Phi
            self.KpPhi, self.KiPhi, self.KdPhi = self.quadTrans.ganhosPhiTrans()
            phi_needed = self.controlePhi(0, self.KpPhi, self.KiPhi, self.KdPhi, self.dt, self.angle[0], 'phiTrans')
            
            # Controle de Theta
            self.KpTheta, self.KiTheta, self.KdTheta = self.quadTrans.ganhosThetaTrans()
            theta_needed = self.controleTheta(0, self.KpTheta, self.KiTheta, self.KdTheta, self.dt, self.angle[1], 'thetaTrans')
            
            # Controle de Psi
            self.KpPsi, self.KiPsi, self.KdPsi = self.quadTrans.ganhosPsiTrans()
            psi_needed = self.controlePsi(0, self.KpPsi, self.KiPsi, self.KdPsi, self.dt, self.angle[2], 'psiTrans')
            
            self.sAltitude = np.array([thrust_needed], dtype=float)
            self.erro_altitude = np.array([self.prev_error], dtype=float)
            
            self.sAtitude =  np.array([phi_needed, theta_needed, psi_needed], dtype=float)
            self.erro_atitude = np.array([self.prev_phi_error, self.prev_theta_error, self.prev_psi_error], dtype = float).reshape(3,1)
            
            U = np.array([thrust_needed, phi_needed, theta_needed, psi_needed], dtype=float)
            self.speeds = self.des2speeds(-U)
            # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}")
            self.lin_acc, self.ang_acc = self.quadTrans.getAccelTrans(self.pos, self.angle, self.ang_vel, self.vel, self.speeds)
            
            
        else:
            
            print("Erro no step")
            print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            
            print(f"Shape das variáveis: pos={np.array(self.pos).shape}, vel={self.vel.shape}, angle={self.angle.shape}, ang_vel={self.ang_vel.shape}, lin_acc={self.lin_acc.shape}, ang_acc={self.ang_acc.shape}")

        # Ajustando os shapes
        # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}, posição: {self.pos}")
        self.pos = np.round(self.pos.reshape(3, 1), 4)
        self.vel = np.round(self.vel.reshape(3, 1), 4)
        self.angle = np.round(self.angle.reshape(3, 1), 4)
        self.ang_vel = np.round(self.ang_vel.reshape(3, 1), 4)
        self.lin_acc = np.round(self.lin_acc.reshape(3, 1), 4)
        self.ang_acc = np.round(self.ang_acc.reshape(3, 1), 4)

        self.vel += self.dt * self.lin_acc
        self.ang_vel += self.dt * self.ang_acc


        self.angle += self.dt * self.ang_vel
        self.angle = self.sat_angle()
        
        # self.angle[2] = self.norm_angle(self.angle[2])
        self.pos += self.dt * self.vel
        self.time += self.dt

    def stepMA(self):
        # Erro de posição
        error = self.ref[2] - self.pos[2]
        self.prev_error = error
        
        # Erro de atitude
        phi_error = 0 - self.angle[0]
        self.prev_phi_error = phi_error
        
        theta_error = 0 - self.angle[1]
        self.prev_theta_error = theta_error
        
        psi_error = 0 - self.angle[2]
        self.prev_psi_error = psi_error
        
        # print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")

        # Determina o ambiente (Ar, Água, ou Transição) com base na altura (pos[2])
        # AR
        if self.pos[2] > 0.25:
            # print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            # print(f"To no Ar {self.pos[2]}")
            self.kT = self.quadAr.kT
            self.K_inv = self.quadAr.K_inv
            self.maxT = self.quadAr.maxT
            self.minT = self.quadAr.minT
            
            self.sAltitude = np.array([0], dtype=float)
            self.sAtitude =  np.array([0, 0, 0], dtype=float)
            
            self.erro_altitude = np.array([self.prev_error], dtype=float)
            
            self.erro_atitude = np.array([self.prev_phi_error, self.prev_theta_error, self.prev_psi_error], dtype = float).reshape(3,1)
            
            U = np.array([0, 0, 0, 0]) # Malha aberta
            self.speeds = self.des2speeds(U)
            # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}")
            self.lin_acc, self.ang_acc = self.quadAr.getAccelAr(self.angle, self.ang_vel, self.speeds)
            
            
        # Agua
        elif self.pos[2] < -0.25:
            # print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            # print(f"To na agua {self.pos[2]}")
            self.kT = self.quadAgua.kT
            # print(self.kT)
            self.K_inv = self.quadAgua.K_inv
            self.maxT = self.quadAgua.maxT
            self.minT = self.quadAgua.minT
            
            self.sAltitude = np.array([0], dtype=float)
            self.sAtitude =  np.array([0, 0, 0], dtype=float)
            self.erro_altitude = np.array([self.prev_error], dtype=float)
            
            self.erro_atitude = np.array([self.prev_phi_error, self.prev_theta_error, self.prev_psi_error], dtype = float).reshape(3,1)
            
            U = np.array([0, 0, 0, 0]) # Malha aberta            
            
            self.speeds = self.des2speeds(U)
            # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}")
            self.lin_acc, self.ang_acc = self.quadAgua.getAccelAgua(self.angle, self.ang_vel, self.vel, self.speeds)
            

        # Transição
        # elif -0.25 <= self.pos[2] <= 0.25:
        elif (-0.25 <= self.pos[2] and self.pos[2] <= 0.25):
            self.vel = np.array([0., 0., 0.], dtype = float).reshape(3,1)
            # print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            # print(f"To na transição {self.pos[2]}")
            # print(f"Ks:{self.Ks}")
            
            self.kT = self.quadAgua.kT
            self.K_inv = self.quadAgua.K_inv
            self.maxT = self.quadAgua.maxT
            self.minT = self.quadAgua.minT

            self.sAltitude = np.array([0], dtype=float)
            self.sAtitude =  np.array([0, 0, 0], dtype=float)
            # self.sAltitude = np.array([thrust_needed], dtype=float)
            self.erro_altitude = np.array([self.prev_error], dtype=float)
            self.erro_atitude = np.array([self.prev_phi_error, self.prev_theta_error, self.prev_psi_error], dtype = float).reshape(3,1)
            
            U = np.array([0, 0, 0, 0]) # Malha aberta
            self.speeds = self.des2speeds(U)
            # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}")
            self.lin_acc, self.ang_acc = self.quadTrans.getAccelTrans(self.pos, self.angle, self.ang_vel, self.vel, self.speeds)

        else:
            
            print("Erro no step")
            print(f"Pos: {self.pos}, Vel: {self.vel}, AngVel: {self.ang_vel}, Accel: {self.lin_acc}, AngAccel: {self.ang_acc}, motores: {self.speeds}")
            
            print(f"Shape das variáveis: pos={np.array(self.pos).shape}, vel={self.vel.shape}, angle={self.angle.shape}, ang_vel={self.ang_vel.shape}, lin_acc={self.lin_acc.shape}, ang_acc={self.ang_acc.shape}")

        # Ajustando os shapes
        # print(f"Speeds: {self.speeds}, aceleração: {self.lin_acc}, posição: {self.pos}")
        self.pos = np.round(self.pos.reshape(3, 1), 4)
        self.vel = np.round(self.vel.reshape(3, 1), 4)
        self.angle = np.round(self.angle.reshape(3, 1), 4)
        self.ang_vel = np.round(self.ang_vel.reshape(3, 1), 4)
        self.lin_acc = np.round(self.lin_acc.reshape(3, 1), 4)
        self.ang_acc = np.round(self.ang_acc.reshape(3, 1), 4)

        self.vel += self.dt * self.lin_acc
        self.ang_vel += self.dt * self.ang_acc


        self.angle += self.dt * self.ang_vel
        self.angle = self.sat_angle()
        # self.angle[2] = self.norm_angle(self.angle[2])
        self.pos += self.dt * self.vel
        # if self.pos[2] < 0:
        #     self.pos[2] = 0
        #     self.vel = np.array([0., 0., 0.], dtype = float).reshape(3,1)
        self.time += self.dt



        