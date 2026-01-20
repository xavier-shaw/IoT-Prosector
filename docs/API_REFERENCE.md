# IoT-Prosector API Reference

Quick reference for all API endpoints in the IoT-Prosector system.

---

## Backend API (Express.js)

**Base URL:** `http://localhost:9990/api`

### Board Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/boards` | Get all boards |
| `GET` | `/boards/:boardId` | Get a specific board by ID |
| `POST` | `/board` | Create a new board |
| `POST` | `/boards/saveBoard` | Save/update a board |

### Function Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/functions` | Get enabled feature toggles |
| `POST` | `/functions` | Update feature toggles |

### State Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/states/:device` | Get all states for a device |

### Video Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/video/get/:id` | Get video by state/action ID |
| `POST` | `/video/upload` | Upload a recorded video |

### Predict Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/predict/:device` | Get prediction for a device |

### Shared Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/shared/start/:stage` | Start sensing for a stage |
| `GET` | `/shared/stop` | Stop sensing |
| `GET` | `/shared/update/:device` | Update device data |

---

## Hardware API (FastAPI)

**Base URL:** `http://localhost:8000`

### System Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | Health check |
| `GET` | `/getBoards` | List all board titles |
| `GET` | `/remove/?device={name}` | Delete all data for a device |
| `GET` | `/removeAllVideos` | Delete all videos |
| `GET` | `/removeAllData` | Delete all sensor data |
| `GET` | `/removeAllBoards` | Delete all boards |

### Port Management

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/getAvailableSensingPorts` | List available serial ports |
| `GET` | `/getConnectedPort` | Get currently connected port |
| `GET` | `/connectPort?port={path}` | Connect to Arduino serial port |

### Sensing Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/startSensing?device={}&idx={}` | Start collecting sensor data |
| `GET` | `/stopSensing` | Stop data collection |
| `GET` | `/storeData?device={}&idx={}` | Save collected data to database |

### Data Processing

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/waitForDataProcessing` | Process raw data (feature extraction, t-SNE, clustering) |
| `POST` | `/loadProcessedData` | Load previously processed data into memory |
| `POST` | `/collage` | Group actions by type |
| `POST` | `/classification` | Generate correlation matrix between clusters and groups |

### Machine Learning

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/train` | Train decision tree classifier |
| `GET` | `/predict` | Predict states using trained model |
| `GET` | `/verify?device={}&predict={}&correct={}` | Record verification results |

---

**Back to [README.md](../README.md)**
