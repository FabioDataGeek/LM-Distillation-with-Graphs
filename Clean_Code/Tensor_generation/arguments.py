def arguments():
    datasets = ['stanfordnlp/sst2']     # 'SetFit/ag_news', 

    return {
        'mode': ['constituency'],
        'datasets': datasets,
        'batch_size': 1,  # Batch size for training
        'model_name': 'google-bert/bert-base-uncased',  # Name of the model to use
        'folders': [f"/usrvol/processed_data/{datasets[0]}"],  # Folders to process
    }