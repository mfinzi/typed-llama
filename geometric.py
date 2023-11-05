#%%
import torch_geometric
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import os
root = os.path.expanduser("~/datasets/QM9")
dataset = QM9(root=root)
#loader = DataLoader(dataset, batch_size=1, shuffle=True)

atomic_symbols = {
    1: 'H',   # Hydrogen
    2: 'He',  # Helium
    3: 'Li',  # Lithium
    4: 'Be',  # Beryllium
    5: 'B',   # Boron
    6: 'C',   # Carbon
    7: 'N',   # Nitrogen
    8: 'O',   # Oxygen
    9: 'F',   # Fluorine
    10: 'Ne'  # Neon
}

# for i, data in enumerate(loader):
#     print(data)
#     print(data.num_graphs)
#     if i >3: break

# print(list(dir(data)))
# print(dataset[0].x)


# from datasets import IterableDataset

# Load the dataset row by row
# construct the output augmented graph
# optionally include the positions


def 

# def text_qm9(root):
#     ds = QM9(root=root)
#     def gen():
#         for idx in range(len(ds)):
#             yield ds[idx]


# def get_molecule_text(row):
#     chars = " ".join([atomic_symbols[int(z.item())] for z in row['charges'] if int(z.item())>0])
#     target = f"{row['homo']:.3f}"
#     return {'text':chars + "\n\n" +target}

# def text_qm9_dataset(tokenizer, split='train',seed=42):
#     torchdss, num_species, charge_scale = QM9datasets()
#     torchds = torchdss[split]
#     def gen():
#         for idx in range(len(torchds)):
#             yield torchds[idx] 
#     ds = IterableDataset.from_generator(gen)
#     ds = ds.map(get_molecule_text)
#     for example in ds:
#         print(example)
#         break
#     ds = ds.map(lambda s: tokenizer(s['text']))
#     ds = ds.shuffle(seed=seed)
#     for example in ds:
#         print(example)
#         break
#     return ds

# %%
