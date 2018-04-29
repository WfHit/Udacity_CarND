## Writeup Report

---

**PID-Controller Project**

---
### Writeup / README

#### 1. Describe the effect each of the P, I, D components had in your implementation.

`Proportional component`: 
Once the system has deviations, proportional component will immediately reduce the deviations. The higher of the value the faster of system responding to errors, but too higher value will make the system unstable.

`Integral component`: 
The integral component can eliminate the effect of system error making the system error zero and finally eliminating the steady-state error.

`Differential component`: 
The differential component, on the one hand, can reduce the overshoot and oscillation, improving the stability of the system, and on the other hand, speed up the dynamic response speed of the system, improving the dynamic performance of the system.

#### 2. Describe how the final hyperparameters were chosen.

First, I use P component only. Starting from a small value, I gradually increase the value of P, until the car oscillating severely. 
Then, I reduce P value a little, and add D component. I adjust the D value so that the car will not oscillate anymore.
Finally, I add I component and tune all three parameter until the car can reduce the error as fast as possible while keeping the oscillation as small as possible.

* P value: 0.155
* I value: 0.000001
* D value: 2.0

