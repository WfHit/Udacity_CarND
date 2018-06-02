## Writeup Report

---

**CarND Path Planning Project Model Documentation**

---
### Writeup / README

[//]: # (Image References)

[system_structure]: ./doc/system_structure.png "system_structure"
[class_diagram]: ./doc/class_diagram.png "class_diagram"
[sequence_chart]: ./doc/sequence_chart.png "sequence_chart"

#### 1. The Overall System Structure

![alt text][system_structure]

As shown in the graph above, the model has two subsystems, the behavioral decision subsystem and the trajectory planning subsystem.  
The left one is behavioral decision subsystem, it's primary responsibility is to decide what ego car should do next, taking surrounding environment satuation into consideration.  
The right one is trajectory planning subsystem, it's primary responsibility is to calculate a smooth path for ego car to follow, according to target pose given by behavioral decision subsystem.
 
#### 2. The Class Diagram

![alt text][class_diagram]

The graph above is the class diagram. The behavioral decision class and trajectory planning class are the two most important class, since the main control logic is implemented in this two class. 
There ars also some other class to help implementing system, and all of them have there own function. Next, Let's discuss them one by one.

`spline`  
This class implements spline interpolation. To use this class, first we need to initialize it with some pairs of data, then we can interpolating with this spline. It's public interface is descripted below:

| Method Name        		|     Description	                              | 
|:---------------------:|:---------------------------------------------:| 
| set_points           	| set points to initialize spline               |
| operator()            | return interpolation at given point           |


`C_Navigation_t`  
This class read in waypoints from file highway_map.csv, then use this data to create three spline object x to s, y to s, dx to s and dy to s. So given s and d we can caculate x, y, dx, dy at any piont. It's public interface is descripted below:

| Method Name         	|     Description	                              | 
|:---------------------:|:---------------------------------------------:| 
| Initialize         	  | create spline of x, y, dx, dy ralate to s     |
| GetSplineXY           | calculate x, y with spline                    |
| GetReferenceSpeed     | return maxmium feasible speed                 | 
| GetTotalLength        | return cyclic length of road                  | 
| GetTotalWidth         | return width of road                          | 


`C_TrafficLane_t`  
This class contains all the cars in this line. With this information, we can calculate the top speed of this line and check whether we are blocked by other cars. It's public interface is descripted below:

| Method Name         	|     Description	                              | 
|:---------------------:|:---------------------------------------------:| 
| AddCarInLane         	| add car to its lane                           |
| ClearCarInLane        | clear cars in this lane                       |
| SetCenterLineFrenetD  | set frenet D of current lane's middle line    |
| GetLaneFrontSpeed     | return the speed to the first front car       | 
| GetTotalWidth         | return width of road                          |	
| GetLaneBackDistance   | return the distance to the first back car     |
| GetCenterLineFrenetD  | return the frenet D of the middle line        |


`C_EgoCar_t`  
This class contains all the information of ego car. It's public interface is descripted below:

| Method Name         	|     Description	                              | 
|:---------------------:|:---------------------------------------------:| 
| UpdateCarInfo         | update ego car state                          |
| SetLane               | update ego car lane                           |
| GetCurrentLane        | return ego car lane                           |
| GetCartesianX         | return ego car x                              | 
| GetCartesianY         | return ego car y                              |	
| GetFrenetS            | return ego car s                              |
| GetFrenetD            | return ego car d                              |
| GetLineSpeed          | return ego car line speed                     |
| GetYawSpeed           | return ego car yaw speed                      |
| GetSamplePeriod       | return sample period                          |
| GetTrajectorySize     | return trajectory size                        |
| SetTrajectorySize     | set trajectory size                           |


`C_BehavioralDecision_t`  
In this class we calculate the target postion ego car should go. Here,I'm not use state machine. In the fact, I use a design pattern called strategy pattern. First, the ego car check the current lane to make sure whether it is blocked by other cars, if no cars in front of ego car, then ego car adopt free go policy. If ego car is blocked by other cars, it will try to find the lowest cost line. If the lowest cost line is other line, then it will adopt change line policy, if the lowest cost line is current line, it will adopt follow car policy. It's public interface is descripted below:

| Method Name         	|     Description	                              | 
|:---------------------:|:---------------------------------------------:| 
| DecideBehavior        | calculate the target postion ego car          |


`C_TrajectoryPlanning_t`  
In this class we calculate plan trajectory according to (start_s end_s) and (start_d end_d) with JMT. It's public interface is descripted below:

| Method Name         	|     Description	                              | 
|:---------------------:|:---------------------------------------------:| 
| PlanTrajectory        | calculate trajectory with JMT                 |


#### 3. System Sequrence Chart

![alt text][sequence_chart]

As shown in system sequrence chart, when system booting up, navigator reads in waypionts and initialize itself. 
When receiving new data, ego car will update its information, and three traffice lane update cars in that lane.
Then it's time to decide next behavior, behavior_decider first check whether current lane is blocked. If not blocked,
ego car will keep in current lane. If current lane is blocked, ego car will try to find the lowest cost lane, 
and change to this lane if it is feasible, and follow front car if it is not feasible.
Finally, ego car plan trajectory with JMT.

#### 4. Reference

Thanks for the great job of salvatorecampagna, [this peoject](https://github.com/salvatorecampagna/CarND/tree/master/term3/project1_path_planning) gives me a lot of inspiration.
