# Installation Guide

This document provides detailed instructions for setting up the IoT-Prosector system on your machine.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Hardware Requirements](#hardware-requirements)
- [Software Installation](#software-installation)
- [Environment Setup](#environment-setup)
- [Running the System](#running-the-system)
---

## Prerequisites

### Required Software Versions

Download [Signoid Hound](https://signalhound.com) and [Arduino](https://www.arduino.cc) on your PC, and have them connected to the hardware.

The following software versions are required for compatibility:

| Software | Required Version | Check Command |
|----------|-----------------|---------------|
| Node.js | v21.2.0 | `node --version` |
| npm | 10.2.3 | `npm --version` |
| Python | 3.10.13 | `python3 --version` |
| pip | Latest | `pip3 --version` |

---

### Hardware Requirements

| Component | Model | Purpose |
|-----------|-------|---------|
| Spectrum Analyzer | Signal Hound SM200 | RF emanation capture |
| Current Sensor | ACS712 (30A module) | Power consumption | 
| Microcontroller | Arduino Uno/Nano | Sensor interface | 
| Network Sniffer | WiFi Pineapple | Traffic capture |

---

## Software Installation

```bash
# Install root JavaScript dependencies
npm install

# Install Backend dependencies
cd Backend && npm install && cd ..

# Install Frontend dependencies
cd Frontend && npm install && cd ..

# Install Hardware (Python/FastAPI) dependencies
cd Hardware

# (Optional) Create and activate Python environment using Anaconda
conda create -n iot-prosector-env python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate iot-prosector-env

# Install Python dependencies
pip install -r requirements.txt

cd ..
```


---

## Environment Setup

You need to create `.env` files in both the `Backend` and `Hardware` directories.

```bash
# MongoDB Configuration
NAME=your_mongodb_username
PASSWORD=your_mongodb_password
CLUSTER=your_cluster.mongodb.net
DB_NAME=iotdb
```
---

## Running the System

From the project root, run:

```bash
npm start
```

This command uses `concurrently` to start all three services simultaneously:
- Backend on http://localhost:9990
- Frontend on http://localhost:5173
- Hardware on http://localhost:8000

---

**Back to [README.md](../README.md)**