import torch
import copy
class Tokenizable(object):
    tokenizer = None
    def tokenize(self, obj:str):
        # returns {input_ids: array of tokens, attention_mask, graph_edges)}
        pass

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


class SingleToken(Tokenizable):
    
    def tokenize(self, obj):
        out_dict = self.tokenizer(obj, add_special_tokens=False)
        assert len(out_dict['input_ids']) == 1, "Assumed\
             to be single token, got {}".format(out_dict['input_ids'])
        out_dict['edges'] = []
        out_dict['prediction_mask'] = [0 for _ in out_dict['attention_mask']]
        return out_dict
# class Sequence(Tokenizable):

class Decimal(Tokenizable):
    def __init__(self, decimals=2, scaler=1.):
        self.decimals = decimals
        self.scaler = scaler
    
    def tokenize(self, x):
        number_string = f"{x/self.scaler:.{self.decimals}f}"
        out_dict = self.tokenizer(number_string, add_special_tokens=False)
        #irange = torch.arange(len(number_string))
        #edges = torch.vstack([irange, irange+1])
        out_dict['edges'] = [(i,i+1) for i in range(len(out_dict['input_ids'])-1)]
        out_dict['prediction_mask'] = [0 for _ in out_dict['attention_mask']]
        return out_dict

    def decode(self, token_ids):
        return float(self.tokenizer.decode(token_ids))*self.scaler

class Sequence(Tokenizable):
    def __init__(self, token_type, sep=None):
        self.token_type = token_type
        self.sep = sep
    
    def tokenize(self, x):
        all_tokenized = self.token_type.tokenize(x[0])
        for xi in x[1:]:
            # add separators
            n = len(all_tokenized['input_ids'])
            if self.sep is not None:
                all_tokenized['input_ids'].append(self.tokenizer.convert_tokens_to_ids(self.sep))
                all_tokenized['attention_mask'].append(1)
                all_tokenized['prediction_mask'].append(0)
            all_tokenized['edges'].append((n-1,n))
            #n = len(all_tokenized['input_ids'])
            xi_tokenized = self.token_type.tokenize(xi)
            j = 1 if self.sep is not None else 0
            all_tokenized['input_ids'].extend(xi_tokenized['input_ids'][j:])
            all_tokenized['attention_mask'].extend(xi_tokenized['attention_mask'][j:])
            all_tokenized['prediction_mask'].extend(xi_tokenized['prediction_mask'][j:])
            all_tokenized['edges'].extend([(ei+n,ej+n) for ei,ej in xi_tokenized['edges'][:]])
        return all_tokenized

    

class Bidirectional(Tokenizable):
    def __init__(self, token_type):
        self.token_type = token_type
    
    def tokenize(self, x):
        all_tokenized = self.token_type.tokenize(x)
        all_tokenized['edges'].extend([(ej,ei) for ei,ej in all_tokenized['edges']])
        # take out duplicates?
        return all_tokenized

class Graph(Tokenizable):
    def __init__(self, token_type, sep=None):
        self.token_type = token_type
        self.sep = sep
        assert sep is None, " not implemented yet"
    
    def tokenize(self, x):
        seq, edges = x
        # here the edges specify how the tokens are connected (whatever they may be)
        # this is operationalized by adding an edge connecting the first token for the token_type tokenization
        all_tokenized = self.token_type.tokenize(seq[0])
        ns = [0] # the indices of the first real token
        for xi in seq[1:]:
            # add separators
            xi_tokenized = self.token_type.tokenize(xi)
            n = len(all_tokenized['input_ids'])
            all_tokenized['input_ids'].extend(xi_tokenized['input_ids'][:])
            all_tokenized['attention_mask'].extend(xi_tokenized['attention_mask'][:])
            all_tokenized['prediction_mask'].extend(xi_tokenized['prediction_mask'][:])
            all_tokenized['edges'].extend([(ei+n,ej+n) for ei,ej in xi_tokenized['edges'][:]])
            ns.append(n)

        additional_edges = [(ns[ei],ns[ej]) for ei, ej in edges]
        #[(i,j) for i in ns for j in ns if i!=j]
        all_tokenized['edges'].extend(additional_edges)
        return all_tokenized

class Tuple(Tokenizable):
    def __init__(self, *token_types, sep=None):
        self.token_types = token_types
        self.sep = sep
    
    def tokenize(self, x):
        all_tokenized = self.token_types[0].tokenize(x[0])
        for xi, token_type in zip(x[1:], self.token_types[1:]):
            n = len(all_tokenized['input_ids'])
            if self.sep is not None:
                all_tokenized['input_ids'].append(self.tokenizer.convert_tokens_to_ids(self.sep))
                all_tokenized['attention_mask'].append(1)
                all_tokenized['prediction_mask'].append(0)
            all_tokenized['edges'].append((n-1,n))
            xi_tokenized = token_type.tokenize(xi)
            j = 1 if self.sep is not None else 0
            all_tokenized['input_ids'].extend(xi_tokenized['input_ids'][j:])
            all_tokenized['attention_mask'].extend(xi_tokenized['attention_mask'][j:])
            all_tokenized['prediction_mask'].extend(xi_tokenized['prediction_mask'][j:])
            all_tokenized['edges'].extend([(ei+n,ej+n) for ei,ej in xi_tokenized['edges'][:]])
        return all_tokenized

class Text(Tokenizable):
    def tokenize(self,x):
        out = self.tokenizer(x, add_special_tokens=False)
        out['edges'] = [(i,i+1) for i in range(len(out['input_ids'])-1)]
        out['prediction_mask'] = [0 for _ in out['attention_mask']]
        return out

class AllPairsTuple(Tokenizable):
    def __init__(self, t1,t2):
        self.token_types = t1,t2
    def tokenize(self,x):
        x1,x2 = x
        t1,t2 = self.token_types
        out = t1.tokenize(x1)
        out2 = t2.tokenize(x2)
        n = len(out['input_ids'])
        out['input_ids'].extend(out2['input_ids'])
        out['attention_mask'].extend(out2['attention_mask'])
        out['prediction_mask'].extend(out2['prediction_mask'])
        out['edges'].extend([(ei+n, ej+n) for ei,ej in out2['edges']])
        all_pairs = [(i,j) for i in range(n) for j in range(n, n+len(out2['input_ids']))]
        out['edges'].extend(all_pairs)
        return out


class Causal(Tokenizable):
    def __init__(self, token_type):
        self.token_type = token_type
    
    def tokenize(self, x):
        out = self.token_type.tokenize(x)
        out['prediction_mask'] = copy.deepcopy(out['attention_mask'])
        return out

import networkx as nx
import matplotlib.pyplot as plt

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

def visualize(tokenized_output):
    G = nx.Graph()
    colors = []
    for i,node in enumerate(tokenized_output['input_ids']):
        l = Tokenizable.tokenizer.decode(node)
        G.add_node(i, label=l)
        if l in atomic_symbols.values():
            colors.append('yellow')
        else:
            colors.append('lightblue')
    G.add_edges_from(tokenized_output['edges'])
    pos = nx.spring_layout(G,iterations=500)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, node_size=100,node_color=colors)
    nx.draw_networkx_edges(G, pos, width=1)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)
    plt.show()
