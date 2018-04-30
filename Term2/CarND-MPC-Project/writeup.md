## Writeup Report

---

**Model Predictive Control Project**

---
### Writeup / README

#### 1. The Model 
Student describes their model in detail. This includes the state, actuators and update equations.  

`State`  
The MPC model defines states as list below:   

| name         		      |     Description	          | 
|:---------------------:|:-------------------------:| 
| x           	        | x position of the car     |
| y                     | y position of the car     |
| psi                 	| orientation of the car    |
| v		            			|	velocity of the car     	|
| cte         	      	| the cross track error     |
| epsi                 	| the orientation error     |

`Actuators`  
The MPC model defines actuators as list below:   

| name         		      |     Description	          | 
|:---------------------:|:-------------------------:| 
| a           	        | throttle                  |
| delta                 | steering angle            |

`Update equations`  
x_[idx+1] = x[idx] + v[idx] * cos(psi[idx]) * dt  
y_[idx+1] = y[idx] + v[idx] * sin(psi[idx]) * dt  
psi_[idx+1] = psi[idx] + v[idx] / Lf * delta[idx] * dt  
v_[idx+1] = v[idx] + a[idx] * dt  
cte[idx+1] = f(x[idx]) - y[idx] + v[idx] * sin(epsi[idx]) * dt  
epsi[idx+1] = psi[idx] - psides[idx] + v[idx] * delta[idx] / Lf * dt  

dt : duration between two timesteps  
Lf : the length from front to CoG  

#### 2. Timestep Length and Elapsed Duration (N & dt)
Student discusses the reasoning behind the chosen N (timestep length) and dt (elapsed duration between timesteps) values. Additionally the student details the previous values tried.

`dt` represents the duration between two timesteps. If this value is high, for example 0.2, the model will lose its accuracy, and as a result, the car tends to oscillate. So this value need to be as low as possible. But during the tuning process, I find when this value is below 0.05, for example 0.01, it is nearly not any help to improve the performance of the system. Finally I choose 0.05.  
`N` represents the timestep length. If this value is low, for example 5, the model predict too near and lost the information of far front,the car tends to leave the road. So this value need to be as high as possible. But if this value is too high, for example 40, its will cost a lot of computations, so I choose value of 20, and as a result the predicting length is roughly equal to the given reference length.  

#### 3. Polynomial Fitting and MPC Preprocessing
A polynomial is fitted to waypoints. & If the student preprocesses waypoints, the vehicle state, and/or actuators prior to the MPC procedure it is described.

The waypionts received from the simulator are in the map coordinate, so I need to transfer them to the car coordinate.(line 109-112 in main.cpp)  
I use 3rd degree polynomial to fit the waypionts, as a result, coeffs[0] is the CTE, coeffs[1] is a funtion of orientation.  

#### 4. Model Predictive Control with Latency
The student implements Model Predictive Control that handles a 100 millisecond latency. Student provides details on how they deal with latency.

Since I choose `dt` to be `0.05 s`, and the latency of simulator is 0.1 s, so I compensante this latency by choose `the second throttle and steering` as the output.(line 157-158 in main.cpp)
