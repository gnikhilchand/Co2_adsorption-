from data_utils import create_dataset, smiles_to_graph
from gnn_model import GCN
from llm_interpreter import LLMInterpreter
from torch_geometric.loader import DataLoader
import torch

# --- 1. Load Data ---
dataset = create_dataset('co2_capture_data.csv')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- 2. Initialize and Train GNN Model ---
# Assuming node features are [atomic_num, degree] -> 2 features
model = GCN(num_node_features=2, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
    return loss

print("Training GNN model...")
for epoch in range(1, 101): # Train for 100 epochs
    loss = train()
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

print("Training complete.")

# --- 3. Run a Prediction and Interpretation ---
model.eval()
# Let's test on a specific molecule from our dataset
test_smiles = "c1ccc(cc1)c1c(O)c(O)c(c(c1O)O)-c1ccccc1"
test_label = 12.7
test_data = smiles_to_graph(test_smiles, test_label)

with torch.no_grad():
    prediction = model(test_data).item()

print(f"\n--- GNN Prediction ---")
print(f"Molecule SMILES: {test_smiles}")
print(f"Actual Value: {test_label}")
print(f"Predicted Value: {prediction:.2f}")


# --- 4. Get LLM Interpretation ---
interpreter = LLMInterpreter()
summary = interpreter.generate_interpretation(test_smiles, prediction)

print("\n--- LLM Generated Summary ---")
print(summary)