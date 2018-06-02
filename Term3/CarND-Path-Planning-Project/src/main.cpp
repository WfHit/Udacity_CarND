#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"

#include "ego_car.hpp"
#include "behavioral_decision.hpp"
#include "trajectory_planning.hpp"

using namespace std;

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

C_EgoCar_t ego_car;  
C_Navigation_t navigator;
C_TrafficLane_t left_lane(&navigator);
C_TrafficLane_t middle_lane(&navigator);
C_TrafficLane_t right_lane(&navigator);
C_BehavioralDecision_t behavior_decider(&left_lane, &middle_lane, &right_lane, &navigator, &ego_car);
C_TrajectoryPlanning_t trajectory_planner(&navigator,&ego_car);
         
int main() {

  uWS::Hub h;
  
  left_lane.SetCenterLineFrenetD(2.0);
  middle_lane.SetCenterLineFrenetD(6.0);
  right_lane.SetCenterLineFrenetD(10.0);

	navigator.Initialize();
	
  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;
  
  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
        	double car_x = j[1]["x"];
        	double car_y = j[1]["y"];
        	double car_s = j[1]["s"];
        	double car_d = j[1]["d"];
        	double car_yaw = j[1]["yaw"];
        	double car_speed = j[1]["speed"];

        	// Previous path data given to the Planner
        	auto previous_path_x = j[1]["previous_path_x"];
        	auto previous_path_y = j[1]["previous_path_y"];
        	// Previous path's end s and d values 
        	double end_path_s = j[1]["end_path_s"];
        	double end_path_d = j[1]["end_path_d"];

        	// Sensor Fusion Data, a list of all other cars on the same side of the road.
        	auto sensor_fusion = j[1]["sensor_fusion"];

        	json msgJson;
        	
        	// TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds
        	
          ego_car.UpdateCarInfo(car_x, car_y, car_s, car_d, car_yaw, car_speed);
          
          //ClearLaneCar;
          left_lane.ClearCarInLane();
          middle_lane.ClearCarInLane();
          right_lane.ClearCarInLane();
      	
          //[ id, x, y, vx, vy, s, d]
          for (int counter=0; counter<sensor_fusion.size(); counter++) {
            CAR_t car;
            car.car_id = sensor_fusion[counter][0];
            car.cartesian_x = sensor_fusion[counter][1];
            car.cartesian_y = sensor_fusion[counter][2];
            car.velocity_x = sensor_fusion[counter][3];
            car.velocity_y = sensor_fusion[counter][4];
            car.frenet_s = sensor_fusion[counter][5];
            car.frenet_d = sensor_fusion[counter][6];
            car.velocity_total = sqrt(car.velocity_x*car.velocity_x + car.velocity_y*car.velocity_y);
            
            if (car.frenet_d>=0.0 && car.frenet_d<4.0) {
              left_lane.AddCarInLane(car);
            } else if (car.frenet_d>=4.0 && car.frenet_d<8.0) {
              middle_lane.AddCarInLane(car);
            } else if (car.frenet_d>=8.0 && car.frenet_d<=12.0) {
              right_lane.AddCarInLane(car);
            } else {
              continue;
            }
          }   	

          if (ego_car.GetFrenetD()>=0.0 && ego_car.GetFrenetD()<4.0) {
            ego_car.SetLane(LEFT_LANE);
          } else if (ego_car.GetFrenetD()>=4.0 && ego_car.GetFrenetD()<8.0) {
            ego_car.SetLane(MIDDLE_LANE);
          } else if (ego_car.GetFrenetD()>=8.0 && ego_car.GetFrenetD()<12.0) {
            ego_car.SetLane(RIGHT_LANE);
          } else {
						ego_car.SetLane(TOTAL_LANES);
          }
           
          double pre_size = previous_path_x.size();					
          vector<double> next_x_vals;
          vector<double> next_y_vals; 
	          
          for(int counter=0; counter<pre_size; counter++)  {
            double prev_x_vals = previous_path_x[counter];
          	double prev_y_vals = previous_path_y[counter];
            next_x_vals.push_back(prev_x_vals);
            next_y_vals.push_back(prev_y_vals);
          }
					
					if(pre_size < 10) {
						behavior_decider.DecideBehavior();
	          trajectory_planner.PlanTrajectory(next_x_vals, next_y_vals);
					}	
					
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

        	auto msg = "42[\"control\","+ msgJson.dump()+"]";

        	//this_thread::sleep_for(chrono::milliseconds(1000));
        	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
