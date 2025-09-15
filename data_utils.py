import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data, DataLoader

# Function to get features for each atom (node)
def get_atom_features(atom):
    # Using simple features: atomic number and number of neighbors
    return [atom.GetAtomicNum(), atom.GetDegree()]

# Main conversion function
def smiles_to_graph(smiles_string, label):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    # Get node features
    atom_features_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features_list, dtype=torch.float)

    # Get edge connections (bonds)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append((i, j))
        edge_indices.append((j, i)) # Edges are undirected

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    # Create the graph data object
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))
    return data

# Function to load and process the whole dataset
def create_dataset(csv_file):
    df = pd.read_csv(csv_file)
    data_list = []
    for index, row in df.iterrows():
        graph = smiles_to_graph(row['smiles'], row['co2_adsorption'])
        if graph:
            data_list.append(graph)
    return data_list

if __name__ == '__main__':
    # Example usage
    dataset = create_dataset('co2_capture_data.csv')
    print(f"Successfully created dataset with {len(dataset)} graphs.")
    print("Example graph:", dataset[1])
    # You can now use this dataset with a PyTorch Geometric DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)