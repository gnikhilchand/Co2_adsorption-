# Hybrid GNN-LLM Framework for CO₂ Capture Material Discovery

This project demonstrates an end-to-end machine learning framework for discovering novel materials for CO₂ capture. It uses a **Graph Neural Network (GNN)** to predict the CO₂ adsorption capacity of materials based on their molecular structure and then leverages a **Large Language Model (LLM)** to interpret these numerical predictions into human-readable summaries.

This approach bridges the gap between complex quantitative predictions and actionable, qualitative insights for researchers.

---

## Features

- **Molecular Graph Representation:** Converts SMILES strings (a text-based representation of molecules) into graph data structures suitable for GNNs using the RDKit library.
- **Predictive Modeling:** Implements a Graph Convolutional Network (GCN) in PyTorch Geometric to perform a regression task, predicting the CO₂ adsorption value for a given material.
- **AI-Powered Interpretation:** Integrates an LLM (e.g., TinyLlama or Mistral-7B) to translate the GNN's numerical output into a natural language summary, classifying the material's potential and providing context.

---

## Technologies Used

- **Programming Language:** Python 3.10+
- **Machine Learning:** PyTorch, PyTorch Geometric
- **LLM Integration:** Hugging Face `transformers`, `accelerate`, `bitsandbytes`
- **Chemoinformatics:** RDKit
- **Data Handling:** Pandas, NumPy

---

## Setup and Installation

Follow these steps to set up the project environment.

### 1. Clone the Repository

First, clone the project repository to your local machine.

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

Create a file named `requirements.txt` in your project folder and paste the following lines into it:

```
pandas
numpy
torch
torchvision
torchaudio
torch_geometric
rdkit-pypi
transformers
accelerate
bitsandbytes
scikit-learn
matplotlib
```

Now, install all the required libraries using pip:

```bash
pip install -r requirements.txt
```

### 4. Authenticate with Hugging Face

To download the LLM, you need to be authenticated with Hugging Face.

```bash
huggingface-cli login
```

Paste your Hugging Face Access Token when prompted.

---

## How to Run the Project

Once the setup is complete, you can run the entire pipeline with a single command:

```bash
python main.py
```

### Expected Output

The script will first train the GNN model, printing the loss at different epochs. Then, it will output the GNN's prediction for a test molecule, and finally, it will display the AI-generated summary from the LLM.

```
Training GNN model...
Epoch: 020, Loss: 76.4255
...
Epoch: 100, Loss: 39.9937
Training complete.

--- GNN Prediction ---
Molecule SMILES: c1ccc(cc1)c1c(O)c(O)c(c(c1O)O)-c1ccccc1
Actual Value: 12.7
Predicted Value: 15.33

--- LLM Generated Summary ---
Based on the predicted CO₂ adsorption value of 15.33 cm³/g, the material with the SMILES string c1ccc(cc1)c1c(O)c(O)c(c(c1O)O)-c1ccccc1 is classified as having **High Potential** for CO₂ capture. This value significantly exceeds the threshold for high-potential materials, indicating strong performance and making it a promising candidate for further experimental investigation.
```

---

## Project Structure

The project is organized into modular Python scripts:

- **`main.py`**: The main entry point of the application. It handles the overall workflow: loading data, training the GNN, making predictions, and calling the LLM for interpretation.
- **`data_utils.py`**: Contains utility functions for data loading and preprocessing. Its primary role is to read the `materials_dataset.csv` and convert the SMILES strings into graph objects for PyTorch Geometric.
- **`gnn_model.py`**: Defines the Graph Convolutional Network (GCN) architecture using PyTorch Geometric layers. This script contains the core deep learning model for prediction.
- **`llm_interpreter.py`**: Manages the interaction with the Hugging Face LLM. It handles loading the model and tokenizer, crafting the prompt, and generating the final text summary.
- **`materials_dataset.csv`**: The dataset containing SMILES strings and their corresponding CO₂ adsorption values.
- **`requirements.txt`**: A list of all Python dependencies required to run the project.
