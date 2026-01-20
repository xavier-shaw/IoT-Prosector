# Walkthrough

This guide walks you through how to use IoT-Prosector to create mental models of IoT devices.

## Table of Contents

- [Getting Started](#getting-started)
- [Stage 1: Interaction](#stage-1-interaction)
- [Stage 2: Data Processing](#stage-2-data-processing)
- [Stage 3: Collage](#stage-3-collage)
- [Stage 4: Verification](#stage-4-verification)
- [Understanding Visualizations](#understanding-visualizations)

---

## Getting Started


Open http://localhost:5173 in your browser.

### 1.. Home Page

The home page displays:
- **Manage Port Connection**: Connect to your Arduino sensing device
- **Manage Functions**: Enable/disable features (recording, visualization, verification)
- **Boards**: List of existing device analysis sessions
- **New Board**: Create a new analysis session

### 2. Connect Sensing Port

Before creating a board:

1. Click **"Manage Port Connection"**
2. Select the Arduino port (e.g., `/dev/ttyUSB0` or `/dev/cu.usbmodem*`)
3. Click **"Confirm"**

### 3. Enable Functions

Click **"Manage Functions"** to enable:

| Function | Description |
|----------|-------------|
| **Recording** | Enable webcam video recording during data collection |
| **Visualization** | Show correlation matrix and scatter plots in Collage stage |
| **Verification** | Enable state prediction testing in Verification stage |

### 4. Create a New Board

1. Click **"New Board"**
2. You'll be redirected to the board workspace
3. Click the title to rename it (e.g., "Google Home Mini")

---

## Stage 1: Interaction

The Interaction stage is where you record device states and the actions that trigger transitions.

### Recording Workflow

#### Step 1: Record Base State

Every analysis chain starts with a **base state** (the device's initial/stable state).

1. Click **"Begin with a new state"**
2. Enter the state name (e.g., "Idle", "Off", "Standby")
3. Click **"Submit"**
4. Click **"Start State Recording"**
5. **Hold the device steady** for ~5 seconds
6. Click **"Confirm"** when the recording dialog appears

The system records:
- Video feed from webcam
- Power consumption data from Arduino
- RF emanations from Signal Hound (if connected)

#### Step 2: Record Action & Next State

After recording a base state:

1. **Choose an action** from the Instruction Table (right panel)
   - Example actions: "Tap", "Voice Command", "Swipe"
2. Enter the **next state name** (e.g., "Playing Music")
3. Click **"Start Action Recording"**
4. **Perform the action** on the device
5. Recording automatically stops after ~2.5 seconds
6. Click **"Confirm"**
7. Click **"Start State Recording"** to record the new state
8. Repeat for more transitions

#### Recording Status Indicators in Code

| Status | Description |
|--------|-------------|
| `start` | Ready to begin a new chain |
| `base state` | Ready to record the base state |
| `choose action` | Select an action from the table |
| `action` | Ready to record the action |
| `state` | Ready to record the state |

### The FSM Panel

As you record, nodes and edges appear on the FSM canvas:

- **Nodes** = Device states (rectangles)
- **Edges** = Transitions with action labels (arrows)

You can:
- **Drag nodes** to rearrange the layout
- **Click nodes** to select and view details
- **Zoom/pan** using scroll and drag

### Saving Progress

Click **"Save"** in the menu bar regularly to persist your work.

### Moving to Collage Stage

When you've recorded sufficient states:

1. Click **"Next"** in the menu bar
2. Wait for data processing (up to 30 seconds)
3. The system extracts features and runs clustering algorithms

---

## Stage 2: Data Processing

When you click **"Next"** after the Interaction stage, the system processes all collected sensor data. This happens automatically in the Hardware API (Python/FastAPI backend).

### Step 1: Fetch Raw Data

For each recorded state, the system retrieves:
- **Power data** from MongoDB (`avg_currents`, `max_currents`, `min_currents`, `times`)
- **Emanation data** from `.pkl` files on disk (FFT results from Signal Hound)

### Step 2: Feature Extraction

#### Power Features
Raw current readings are grouped into windows and statistical features are calculated:
- Groups of 8 consecutive readings → calculate **mean** for each group
- Each state produces ~25 power feature samples

#### Emanation Features
For each FFT sweep, the system calculates 11 statistical features:

| Feature | Description |
|---------|-------------|
| Mean | Average power level |
| Median | Middle value |
| Std | Standard deviation |
| Variance | Spread of values |
| RMS | Root mean square |
| MAD | Median absolute deviation |
| Skewness | Asymmetry of distribution |
| Kurtosis | Tail heaviness |
| IQR | Interquartile range |
| MSE | Mean squared error from mean |

Emanation data is normalized (min-max scaling) and averaged into 25 samples per state.

### Step 3: Combine Features

Power and emanation features are concatenated into a single feature vector:

```
Feature Vector = [power_features (8 dims)] + [emanation_features (11 dims)]
                = 19-dimensional vector per sample
```

### Step 4: Dimensionality Reduction (t-SNE)

The high-dimensional feature vectors are reduced to 2D using **t-SNE** (t-distributed Stochastic Neighbor Embedding):
- Preserves local structure (similar states stay close)
- Parameters: `perplexity=5`, `learning_rate='auto'`

**Output 2D coordinates for visualization in scatterplot.**

### Step 6: K-Means Clustering

The system automatically discovers natural groupings in the data:

1. **Test multiple cluster counts** (2 to N states)
2. **Calculate silhouette score** for each — measures cluster quality
3. **Select best K** — highest silhouette score
4. **Assign each data point** to a cluster
5. **Build distribution dict** — for each state, count how many samples fall into each cluster

**Output the cluster distribution per state that can be used for correlation matrix.**

---

## Stage 3: Collage

The Collage stage allows you to group similar states to build a clear state machine or mental model of the IoT device.

### Auto-Clustering

1. Click the **"Collage"** button at the top
2. The system automatically groups similar states based on sensor data
3. States with similar power/emanation patterns are grouped together

### Manual Grouping

If you do not want to use auto-clustering, you can manually cluster the nodes:

1. **Drag a state node** onto another state
2. They merge into a **semantic group** (larger container node)
3. The group represents states that you consider functionally equivalent

### Using Visualizations

Toggle between visualizations to guide your grouping decisions:

#### Correlation Matrix
Shows the relationship between:
- **Rows**: Clusters from K-Means (Sensing Model)
- **Columns**: Your annotated groups (Mental Model)

Higher values (darker colors) indicate stronger correlation.

#### Distribution Scatterplot
Shows t-SNE embeddings of sensor data:
- Each point is a data sample
- Colors represent states/groups
- Closer points = similar sensor signatures

Click on a state in the FSM panel to highlight its points.

### Preview Final FSM

Click **"Preview & Annotate"** to see how the grouped FSM will look.

### Moving to Verification Stage

1. Click **"Next"** in the menu bar
2. Wait for model training (up to 30 seconds)
3. The system trains a decision tree classifier

---

## Stage 4: Verification

The Verification stage tests whether the model can correctly predict device states.

### Testing Predictions

1. The system shows the trained model's predictions
2. For each state in your test set:
   - The **predicted state** is highlighted on the FSM
   - Compare with the **actual state** you recorded
3. Click to confirm or correct predictions

### Validation Workflow

1. **Interact with the device** to trigger a state change
2. The system captures new sensor data
3. The model predicts which state the device is in
4. You **confirm or correct** the prediction
5. Corrections improve future iterations

### Finishing

Click **"Next"** to return to the home page with your completed analysis.

---

## Understanding Visualizations

### Correlation Matrix

**Interpretation:**
- Values range from 0 to 1
- High diagonal values = good alignment between sensing and mental models
- Off-diagonal values = potential confusion between states

### t-SNE Scatterplot

**Interpretation:**
- Clusters that are far apart are easily distinguishable
- Overlapping clusters may indicate states that are hard to differentiate
- Outliers may indicate recording errors or transitional states

---

**Back to [README.md](../README.md)**