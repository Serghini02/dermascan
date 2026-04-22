# 🔬 DermaScan — Intelligent Mole Scanner

AI-powered application for dermatological mole analysis using Computer Vision, Deep Learning (CNN), Reinforcement Learning (DQN), and Natural Language Processing (NLP).

---

## 📋 Prerequisites

- [Docker](https://www.docker.com/get-started) and Docker Compose installed
- Python 3.11+ (only for local execution without Docker)
- Internet access (for the first image build)

---

## 🐳 Startup with Docker (Recommended)

### 1. Clone the repository

```bash
git clone <repository-url>
cd FINAL_Dermascan
```

### 2. Build and start the container

```bash
docker compose up --build -d
```

The first time may take a few minutes as it downloads TTS models and dependencies.

### 3. Access the application

```
http://localhost:3333
```

### 4. View real-time logs

```bash
docker compose logs -f
```

### 5. Stop the application

```bash
docker compose down
```

---

## 🖥️ Local Startup (without Docker)

### 1. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the Flask server

```bash
python app.py
```

The application will be available at:

```
http://localhost:5000
```

> ⚠️ In local mode, the HAM10000 dataset is automatically downloaded from Kaggle if not available locally. Requires Kaggle credentials to be configured.

---

## 🖧 Deployment on SSH Server (Production)

The project is designed to run on a Linux server with Docker.

### 1. Connect to the server

```bash
ssh sergio@ariarca
```

### 2. Navigate to the project directory

```bash
cd stack/dermascan
```

### 3. Start / Restart

```bash
# Start for the first time or after Dockerfile changes
docker compose up --build -d

# Restart without rebuilding (after code changes)
docker compose restart

# View status
docker compose ps

# View logs
docker compose logs -f
```

### 4. Update files individually (without rebuild)

If only static files or Python files are modified (no changes in `requirements.txt` or `Dockerfile`), you can copy them directly and restart:

```bash
# From your local machine
scp static/js/app.js sergio@ariarca:stack/dermascan/static/js/app.js
scp templates/index.html sergio@ariarca:stack/dermascan/templates/index.html

# Restart on the server
ssh sergio@ariarca "cd stack/dermascan; docker compose restart"
```

The production application is exposed on **port 3333** (mapped internally to Flask's 5000).

---

## 🗂️ Project Structure

```
FINAL_Dermascan/
├── app.py                  # Main Flask server
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image
├── docker-compose.yml      # Docker orchestration
├── README.md               # This file
├── database/
│   └── db_manager.py       # SQLite management
├── models/                 # Trained models (.pth)
├── nlp/                    # NLP and correction modules
├── rl/                     # DQN Agent (Reinforcement Learning)
├── vision/
│   ├── cnn_model.py        # Fine-tuned EfficientNet-B1
│   ├── skin_analyzer.py    # ABCDE analysis (OpenCV)
│   ├── expert_system.py    # Expert System (Medical rules)
│   └── metaheuristic_tuner.py  # Genetic Algorithm (Optimization)
├── static/
│   ├── css/style.css
│   └── js/app.js
└── templates/
    └── index.html
```

---

## 🤖 Included AI Modules

| Module | Technology | Description |
|---|---|---|
| **CNN** | EfficientNet-B1 (PyTorch) | Classification of 7 lesion types (HAM10000) |
| **ABCDE** | OpenCV | Visual morphological analysis of moles |
| **DRL** | DQN (PyTorch) | Agent that decides which questions to ask |
| **NLP** | Tokenization + Regex | Extraction of symptoms from patient text |
| **Expert System** | Medical Rules | Final diagnosis refinement |
| **Metaheuristic** | Genetic Algorithm | Model hyperparameter optimization |

---

## ⚙️ Environment Variables

| Variable | Default Value | Description |
|---|---|---|
| `FLASK_ENV` | `production` | Server mode |
| `SKIP_DATASET_DOWNLOAD` | `true` | Skip HAM10000 download at startup |
| `SECRET_KEY` | `dermascan-prod-2026` | Flask secret key |

---

## 🔧 Common Troubleshooting

**The container doesn't start:**
```bash
docker compose logs dermascan
```

**CNN model not available:**
Ensure `models/skin_cnn.pth` exists. If not, the system will work without CNN classification, but other modules will remain operational.

**Application doesn't load changes after updating files:**
Force reload in the browser with `Ctrl + Shift + R` or open in incognito mode.
