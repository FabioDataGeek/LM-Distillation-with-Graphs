from torch.utils.data import Dataset, DataLoader


""" 
With this dataloader we are going to generate the different graphs from sentences based on the format of 
each dataset and the configuration of the graphs we want to generate.
"""

class SNLIDataset(Dataset):
    """
    Dataset class for generating graphs from sentences.
    """
    
    def __init__(self, sentences: list):
        """
        Initialize the dataset with a list of sentences and parsers.
        """
        self.sentences = sentences
        
    def __len__(self):
        """
        Return the number of sentences in the dataset.
        """
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Return the syntactic, semantic, and constituency graphs of the sentence at the given index.
        """
        return self.sentences[idx]['sentence1'], self.sentences[idx]['sentence2'], self.sentences[idx]['gold_label']



class RTEDataset(Dataset):
    """
    Dataset class for generating graphs from sentences.
    """
    
    def __init__(self, sentences: list):
        """
        Initialize the dataset with a list of sentences and parsers.
        """
        self.sentences = sentences
        
    def __len__(self):
        """
        Return the number of sentences in the dataset.
        """
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Return the syntactic, semantic, and constituency graphs of the sentence at the given index.
        """
        return self.sentences[idx]['sentence1'], self.sentences[idx]['sentence2'], self.sentences[idx]['label']
    


class SciTailDataset(Dataset):
    """
    Dataset class for generating graphs from sentences.
    """
    
    def __init__(self, sentences: list):
        """
        Initialize the dataset with a list of sentences and parsers.
        """
        self.sentences = sentences
        
    def __len__(self):
        """
        Return the number of sentences in the dataset.
        """
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Return the syntactic, semantic, and constituency graphs of the sentence at the given index.
        """
        return self.sentences[idx]['sentence1'], self.sentences[idx]['sentence2'], self.sentences[idx]['gold_label']