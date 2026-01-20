# IoT-Prosector Architecture

This document provides a comprehensive overview of the IoT-Prosector system architecture, including its components, data flow, and design decisions.

## Table of Contents

- [System Overview](#system-overview)
- [Component Details](#component-details)
  - [Frontend](#frontend)
  - [Backend](#backend)
  - [Hardware](#hardware)
- [Technology Stack](#technology-stack)

---

## System Overview

IoT-Prosector is an interactive system designed to help users form mental models of black-box IoT devices. It achieves this through:

1. **Multi-modal Sensing**: Combines power consumption, network traffic, and radio emanations to capture device behavior
2. **Finite State Machine (FSM) Visualization**: Represents IoT device states and transitions as an interactive graph
3. **Machine Learning Pipeline**: Clusters sensor data and trains classifiers to predict device states

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           IoT-Prosector System                             │
│                                                                            │
│  ┌─────────────────────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │        Hardware             │     │   Backend   │     │   Frontend  │   │
│  │        (FastAPI)            │◄───►│ (Express.js)│◄───►│  (React.js) │   │
│  │        Port 8000            │     │  Port 9990  │     │  Port 5173  │   │
│  └──────────────┬──────────────┘     └──────┬──────┘     └─────────────┘   │
│                 │                           │                              │
│                 ▼                           ▼                              │
│         ┌───────────────┐            ┌─────────────┐                       │
│         │ Sensing       │            │  MongoDB    │                       │
│         │ Devices       │            │  Atlas      │                       │
│         └───────────────┘            └─────────────┘                       │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Frontend

**Location**: `/Frontend/`  
**Port**: 5173 (Vite dev server)  
**Framework**: React.js with Vite

The frontend provides an interactive visualization interface built with React Flow for FSM manipulation.

#### Directory Structure

```
Frontend/
├── src/
│   ├── App.jsx                 # Main application router
│   ├── containers/
│   │   ├── home.jsx            # Home page - board management
│   │   └── board.jsx           # Main workspace - FSM editor
│   ├── components/
│   │   ├── NodeChart.jsx       # React Flow FSM visualization
│   │   ├── MenuBar.jsx         # Navigation and actions
│   │   ├── InteractionRecorder.jsx  # State/action recording UI
│   │   ├── CollagePanel.jsx    # Clustering visualization
│   │   ├── VerificationPanel.jsx    # Model testing UI
│   │   ├── InstructionTable.jsx     # Interaction guide
│   │   ├── *Node.jsx           # Various node components
│   │   └── *Edge.jsx           # Various edge components
│   └── shared/
│       ├── routes.js           # Route definitions
│       └── chartStyle.js       # Visual styling constants
└── public/                     # Static assets (interaction images)
```

#### Key Components

| Component | Purpose |
|-----------|---------|
| `NodeChart` | React Flow canvas for FSM visualization and editing |
| `InteractionRecorder` | Controls for starting/stopping sensor data recording |
| `CollagePanel` | Displays correlation matrix and t-SNE scatter plots |
| `VerificationPanel` | Interface for testing trained model predictions |
---

### Backend

**Location**: `/Backend/`  
**Port**: 9990  
**Framework**: Express.js with Mongoose

The backend serves as the data persistence layer, managing boards, states, and videos in MongoDB.

#### Directory Structure

```
Backend/
├── server.js                   # Express app entry point
└── api/
    ├── controllers/
    │   ├── boardController.js  # Board CRUD operations
    │   ├── stateController.js  # State management
    │   ├── videoController.js  # Video storage
    │   ├── predictController.js # Prediction results
    │   └── sharedController.js # Shared utilities
    ├── models/
    │   ├── boardModel.js       # Board schema (FSM + metadata)
    │   ├── stateModel.js       # State schema
    │   ├── videoModel.js       # Video schema
    │   ├── predictModel.js     # Prediction schema
    │   └── sharedModel.js      # Shared data schema
    ├── routes/
    │   └── *Routes.js          # Route definitions
    └── util/
        └── constants.js        # Configuration constants
```
---

### Hardware

**Location**: `/Hardware/`  
**Port**: 8000  
**Framework**: FastAPI (Python)

The hardware module interfaces with sensing devices and performs ML processing.

#### Directory Structure

```
Hardware/
├── main.py                 # FastAPI app with all endpoints
├── power_data.py           # Arduino serial communication
├── emanation_data.py       # Signal Hound data collection
├── network_data.py         # Network traffic capture
├── FFTpeaks.py             # FFT analysis for emanations
├── acs712.ino              # script for Arduino current sensor
├── getEmanations_2.sh      # Shell script for Signal Hound
└── requirements.txt        # Python dependencies
```
---

## Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Frontend** | React.js | 18.x | UI Framework |
| | Vite | 4.x | Build tool |
| | React Flow | 11.x | Graph visualization |
| | MUI | 5.x | UI components |
| | Axios | 1.x | HTTP client |
| **Backend** | Node.js | 21.2.0 | Runtime |
| | Express.js | 4.x | Web framework |
| | Mongoose | 7.x | MongoDB ODM |
| **Hardware** | Python | 3.10.13 | Runtime |
| | FastAPI | 0.100+ | Web framework |
| | scikit-learn | 1.x | ML algorithms |
| | NumPy/Pandas | - | Data processing |
| | PySerial | 3.x | Arduino communication |
| **Database** | MongoDB Atlas | - | Cloud database |
| **Sensing** | Arduino | - | Power sensing |
| | Signal Hound SM200 | - | RF emanation capture |

---


**Back to [README.md](../README.md)**