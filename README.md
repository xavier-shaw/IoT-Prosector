# IoT-Prosector: Reasoning about the Internal States of Blackbox IoT Devices Using Side-Channel Information

IoT-Prosector is an interactive system designed to help users form mental models of black-box IoT devices. It consists of two key components: (i) a multi-modal sensing technique that combines power consumption, network traffic, and radio emanations; (ii) an annotation interface with interactive visualizations, enabling users to construct and refine these mental models as finite state machines.

### Installation

First clone this repository:
```
git clone https://github.com/your-org/IoT-Prosector.git
```

Then refer to [Installation Guide](/docs/INSTALLATION.md) for detailed guidance for installation.

### Running

IoT-Prosector contains a python backend for hardware connection, a express.js backend for data management, and a react frontend for GUI.

To run IoT-Prosector, you could simply use the command: `npm start` at the root of this directory.

Behind the scene, this command setup the Frontend (http://localhost:5173/), the Data backend (http://localhost:9990/), and Hardware backend (http://localhost:8000/) simultanously.

## How to Use

IoT-Prosector guides you through a **three-stage workflow** to build a mental model of your IoT device:

### Stage 1: Interaction — *Record device behaviors*

1. **Connect** your sensing hardware (Arduino + Signal Hound)
2. **Create a new board** for your IoT device
3. **Record states**: Put the device in a stable state → Click "Start State Recording" → Hold for 5 seconds
4. **Record transitions**: Select an action → Click "Start Action Recording" → Perform the action on the device
5. **Repeat** to capture all device states and transitions

### Stage 2: Collage — *Build the Mental Model (or State Machine) of the device*

1. Click **"Next"** to trigger ML processing (clustering + dimensionality reduction)
2. Review the **correlation matrix** showing how sensor patterns map to your labeled states
3. **Drag and drop** states on the FSM canvas to group ones that behave similarly
4. Use the **scatterplot** to visualize sensor data distribution

### Stage 3: Verification — *Test your model*

1. Click **"Next"** to train a classifier on your grouped states
2. **Interact** with the device in real-time
3. The system **predicts** which state the device is in
4. **Confirm or correct** predictions to validate your mental model

---

For detailed step-by-step instructions, see the [Walkthrough](/docs/WALKTHROUGH.md).   
