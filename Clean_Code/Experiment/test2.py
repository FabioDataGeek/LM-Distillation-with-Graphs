import pickle as pkl

with open('/usrvol/processed_tensors/SNLI/dev/semantic/bert-base-uncased/raw/semantic0.pkl', 'rb') as f:
    data = pkl.load(f)


graph1, graph2, label = data[0]


print("")

