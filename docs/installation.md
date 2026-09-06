# Installation and Setup Guide — NeuroVR / STAAR

Follow these steps to set up and run the NeuroVR / STAAR project on macOS, Linux, or Windows.

---

## 1. Prerequisites

- **Python:** Python 3.11 (Python 3.11.16 recommended).
- **Git:** Standard Git version control.
- **Hardware Acceleration:**
  - **macOS:** Apple Silicon (M1/M2/M3/M4) supported natively via Metal Performance Shaders (MPS).
  - **Linux / Windows:** NVIDIA GPU with CUDA supported, or CPU fallback.

---

## 2. Step-by-Step Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd major-project
```

### Step 2: Create a Python Virtual Environment
```bash
python3.11 -m venv .venv
```

Activate the environment:
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (CMD / PowerShell):**
  ```powershell
  .venv\Scripts\activate
  ```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment (Optional)
Copy the example environment file if you wish to customize ports or credentials:
```bash
cp .env.example .env
```

---

## 3. Verifying Installation

Run the project inputs verification script to confirm all dependencies and medical imaging tools are functional:
```bash
python scripts/validate_project_inputs.py
```

Expected output:
```text
✓ PROJECT INPUTS VALID
```
