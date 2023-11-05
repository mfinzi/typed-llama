from corm_data import ProcessedDataset, collate_fn, initialize_datasets
import os
import torch

default_qm9_dir = '~/datasets/molecular/qm9/'
def QM9datasets(root_dir=default_qm9_dir):
    root_dir = os.path.expanduser(root_dir)
    filename= f"{root_dir}data.pz"
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        datasets, num_species, charge_scale = initialize_datasets((-1,-1,-1),
         "data", 'qm9', subtract_thermo=True,force_download=True)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)
            dataset.num_species = 5
            dataset.charge_scale = 9
        os.makedirs(root_dir, exist_ok=True)
        torch.save((datasets, num_species, charge_scale),filename)
        return (datasets, num_species, charge_scale)
    
datasets, num_species, charge_scale = QM9datasets()

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

from datasets import IterableDataset

def get_molecule_text(row):
    chars = " ".join([atomic_symbols[int(z.item())] for z in row['charges'] if int(z.item())>0])
    target = f"{row['homo']:.3f}"
    return {'text':chars + "\n\n" +target}

def text_qm9_dataset(tokenizer, split='train',seed=42):
    torchdss, num_species, charge_scale = QM9datasets()
    torchds = torchdss[split]
    def gen():
        for idx in range(len(torchds)):
            yield torchds[idx] 
    ds = IterableDataset.from_generator(gen)
    ds = ds.map(get_molecule_text)
    for example in ds:
        print(example)
        break
    ds = ds.map(lambda s: tokenizer(s['text']))
    ds = ds.shuffle(seed=seed)
    for example in ds:
        print(example)
        break
    return ds



# print(datasets['train'][0]['charges'],datasets['train'][0]['homo'], datasets)