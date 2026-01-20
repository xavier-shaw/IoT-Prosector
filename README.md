# IoT-Prosector: Reasoning about the Internal States of Blackbox IoT Devices Using Side-Channel Information

[📖 See the arXiv paper of IoT-Prosector.](https://arxiv.org/abs/2311.13761)

[🎥 Watch a quick demo video of IoT-Prosector in action.](https://drive.google.com/file/d/11PI1Mzo8_sQ1wGWKhMIwUCWfK3JaIUJN/view?usp=sharing)

IoT-Prosector is an interactive system designed to help users form mental models of black-box IoT devices. It consists of two key components: (i) a multi-modal sensing technique that combines power consumption, network traffic, and radio emanations; (ii) an annotation interface with interactive visualizations, enabling users to construct and refine these mental models as finite state machines.


![IoT-Prosector Teaser Figure](/docs/figures/IoT%20Setup.jpg)

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](/docs/ARCHITECTURE.md) | System design and component overview |
| [API Reference](/docs/API_REFERENCE.md) | Backend and Hardware API endpoints |
| [Installation](/docs/INSTALLATION.md) | Setup and configuration guide |
| [Walkthrough](/docs/WALKTHROUGH.md) | Detailed usage instructions |

## Quick Start

### Installation

First clone this repository:
```
git clone https://github.com/your-org/IoT-Prosector.git
```

Then refer to [Installation Guide](/docs/INSTALLATION.md) for detailed guidance for installation.

### Running

IoT-Prosector contains a python backend for hardware connection, a express.js backend for data management, and a react frontend for GUI. You can refer to [Architecture](/docs/ARCHITECTURE.md) to see details of the system design. 

To run IoT-Prosector, you could simply use the command: `npm start` at the root of this directory.

Behind the scene, this command setup the Frontend (`http://localhost:5173/`), the Data backend (`http://localhost:9990/`), and Hardware backend (`http://localhost:8000/`) simultanously.

## How to Use

![IoT-Prosector Workflow](/docs/figures/IoT%20Workflow.jpg)

IoT-Prosector guides you through a **four-stage workflow** to build a mental model of your IoT device:

### Stage 1: Interaction — *Record device behaviors*

1. **Connect** your sensing hardware (Arduino + Signal Hound)
2. **Create a new board** for your IoT device
3. **Record states**: Put the device in a stable state → Click "Start State Recording" → Hold for 5 seconds
4. **Record transitions**: Select an action → Click "Start Action Recording" → Perform the action on the device
5. **Repeat** to capture all device states and transitions

### Stage 2: Modeling — *Process sensor data*

When you click **"Next"**, the system automatically:
1. **Extracts features** from power readings and RF emanation data
2. **Reduces dimensions** using t-SNE to project data into the 2D space
3. **Clusters data** using K-Means to find natural groupings in sensor patterns

This processing takes ~30 seconds and prepares the data for the Collage stage.

### Stage 3: Collage — *Build the Mental Model (or State Machine) of the device*

1. Click **"Next"** to trigger ML processing (clustering + dimensionality reduction)
2. Review the **correlation matrix** showing how sensor patterns map to your labeled states
3. **Drag and drop** states on the FSM canvas to group ones that behave similarly
4. Use the **scatterplot** to visualize sensor data distribution

### Stage 4: Verification — *Test your model*

1. Click **"Next"** to train a classifier on your grouped states
2. **Interact** with the device in real-time
3. The system **predicts** which state the device is in
4. **Confirm or correct** predictions to validate your mental model

---

For detailed step-by-step instructions, see the [Walkthrough](/docs/WALKTHROUGH.md).   