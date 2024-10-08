# IoT Prosector

IoTProsector: Reasoning about the Internal States of Blackbox IoT Devices Using Side-Channel Information

IoTProsector is an interactive system designed to help users form mental models of black-box IoT devices. It consists of two key components: (i) a multi-modal sensing technique that combines power consumption, network traffic, and radio emanations; (ii) an annotation interface with interactive visualizations, enabling users to construct and refine these mental models as finite state machines.

## Setup

### Installation

First, download [Signoid Hound](https://signalhound.com) and [Arduino](https://www.arduino.cc) on your PC, and have them connected to the hardware.

Then, install the dependencies of all three modules (i.e., Backend, Frontend, and Hardware) by running following commands:

```
npm install
cd Backend && npm install
cd Frontend && npm install
cd Hardware && pip install -r requirements.txt
```

Since IoTProsector uses MongoDB for data storage, you should create a `.env` file under both Backend and Hardware before running. The `.env` file should include `NAME`, `PASSWORD`, `CLUSTER`, and `DB_NAME`.

The versions of the engines I used during development are as follows:
 
```
"node": "v21.2.0"
"npm": "10.2.3"
"python": "3.10.13" 
```
### Running

To run IoTProsector, you could simply use the command: `npm start` and open http://localhost:5173/.

Behind the scene, this command actually setup the Backend (http://localhost:9990/), Frontend (http://localhost:5173/), and Hardware (http://localhost:8000/) simultanously.

## How to Use

### Preparation
Before creating a new board for the IoT device, choose the port that connected with the sensing device. Also, you can enable or disable functions based on your preference.

### Finite State Machine
Given each IoT device has a finite set of states and possible transitions between states, a Finite State Machine (FSM) can be used to effectively represent the IoT device’s states and the transition events between states. Specifically, each node in the FSM represents the IoT state and the edge connecting two nodes represents the transition event.

### Exploring

The first step to prosect an IoT device is to explore its possible states as many as possible. To support this, our system incorporate a recording module.
It helps user to record videos and sensing data of IoT devices during stable state and transition (i.e., when perform an interaction).

When the device is at a stable state, user should click "Start State Recording" button. In other case, before interacting with the device, user should select an interaction that is to be doen, then click "Start Action Recording" button. 

### Modeling

After collecting sensing data, IoTProsector would develop a sensing model which utilizes clustering algorithms to distinguish data points that belong to different IoT device's states. This modeling process would be triggered by clicking "Next" button at "Exploration" stage, and it takes some time to perform the calculations.

### Collaging

To further prosect the states of the IoT device, our system introduces an interactive visualization module to aid users better understand the IoT device at a more grained level.

Users can drag and drop the states on the FSM panel to form a `group state` if they find some states share the same meaning. At the same time, our system provides: (1) a correlation matrix which illustrates the relationship between clusters generated by the sensing model and states annotated by users; (2) a scatterplot which demonstrates the distribution of sensor data and the relationship between each data point and its corresponding annotated state.   
