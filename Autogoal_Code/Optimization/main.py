import torch
from dig.xgraph.method import SubgraphX
import numpy as np
from tqdm import tqdm
from experiment.Optimization.architecture_GNNs_single import *
from experiment.Clustered_optimization.dataloader import *
import pickle as pkl
from datetime import datetime
from autogoal.kb import AlgorithmBase, SemanticType
from autogoal.grammar import (
    DiscreteValue,
    ContinuousValue,
)
from autogoal.utils import nice_repr

"""HIPERPARÁMETROS A OPTIMIZAR"""

# num_hops = 1  # int
# rollout = 50  # int
# min_atoms = 2  # int
# c_puct = 10  # int
# expand_atoms = 2  # int
# local_radius = 1  # int
# sample_num = 5  # int
# max_nodes = 15  # int


def explain(
    algorithm,
    dataset,
    device,
    num_hops,
    rollout,
    min_atoms,
    c_puct,
    expand_atoms,
    local_radius,
    sample_num,
    max_nodes,
):
    explainer = SubgraphX(
        model=algorithm.model,
        num_classes=args['labels'],
        device=device,
        num_hops=num_hops,
        rollout=rollout,
        min_atoms=min_atoms,
        c_puct=c_puct,
        expand_atoms=expand_atoms,
        local_radius=local_radius,
        sample_num=sample_num,
        save_dir="/usrvol/explainability_results/",
    )
    masked_list = []
    maskout_list = []
    sparsity_list = []
    loader = HomogeneousDataLoader(dataset, batch_size=1, shuffle=False)
    
    for batch in tqdm(loader):
        graph, label = batch
        # Graphs input information for subgraphX
        data = Data(x=graph.x, edge_index=graph.edge_index, batch= graph.batch).to(device)
        x = data.x
        edge_index = data.edge_index
        label = int(label)
        batch = data.batch
        
        _, related_pred = explainer.explain(
            x=x,
            edge_index=edge_index,
            label=label,
            max_nodes=max_nodes,
        )
        masked_list.append(related_pred["masked"])
        maskout_list.append(related_pred["maskout"])
        sparsity_list.append(related_pred["sparsity"])
        

    # Estos tres valores los queremos maximizar
    return (masked_list, maskout_list, sparsity_list)


def load_dataset():
    dataset = Dataset_GNN(
        root=args["root_test_data_path"], files_path=args["raw_test_data_path"]
    )
    return dataset


class XGraph(SemanticType):
    pass


class GraphSeqExplanation(SemanticType):
    pass

@nice_repr
class SubgraphXAutogoal(AlgorithmBase):
    def __init__(
        self,
        num_hops: DiscreteValue(1, 5),  # type: ignore
        rollout: DiscreteValue(50, 300),  # type: ignore
        min_atoms: DiscreteValue(1, 10),  # type: ignore
        c_puct: ContinuousValue(1, 30),  # type: ignore
        expand_atoms: DiscreteValue(1, 5),  # type: ignore
        local_radius: DiscreteValue(1, 5),  # type: ignore
        sample_num: DiscreteValue(1, 5),  # type: ignore
        max_nodes: DiscreteValue(2, 40),  # type: ignore
    ):
        self.model = GNN_classifier(
            size=args["size"],
            num_layers=args["num_layers"],
            dropout=args["dropout"],
            module=args["module"],
            layer_norm=args["layer_norm"],
            residual=args["residual"],
            pooling=args["pooling"],
            lin_transform=args["lin_transform"],
        )
        self.state_dict = torch.load(args['model_dir'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_hops = num_hops
        self.rollout = rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.local_radius = local_radius
        self.sample_num = sample_num
        self.max_nodes = max_nodes

    def load_model(self):
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()

    def save_results(self, masked_score, maskout_score, sparsity_score):
        results = {
            "masked_score": masked_score,
            "maskout_score": maskout_score,
            "sparsity_score": sparsity_score,
        }
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(
            f"/usrvol/explainability_results/results{current_datetime}.pkl", "wb"
        ) as f:
            pkl.dump(results, f)

    def run(self, dataset: XGraph) -> GraphSeqExplanation:
        self.load_model()
        return explain(
            self,
            dataset,
            self.device,
            self.num_hops,
            self.rollout,
            self.min_atoms,
            self.c_puct,
            self.expand_atoms,
            self.local_radius,
            self.sample_num,
            self.max_nodes,
        )

def my_metric(X, result):
    masked_score = np.mean(result[0])
    maskout_score = np.mean(result[1])
    sparsity_score = np.mean(result[2])
    return np.mean([masked_score, maskout_score, sparsity_score])


def run_autogoal():
    # AutoGOAL Example: basic usage of the AutoML class
    from autogoal.kb import MatrixContinuousSparse, VectorDiscrete
    from autogoal.ml import AutoML, calinski_harabasz_score
    from autogoal.utils import Min, Gb, Hour, Sec
    from autogoal.search import PESearch, JsonLogger, ConsoleLogger
    from autogoal.utils._process import initialize_cuda_multiprocessing

    initialize_cuda_multiprocessing()

    # Load dataset
    initial_dataset = load_dataset()
    np.random.seed(54)
    sample = np.random.choice(args['dataset_length'], args['num_samples'])
    training_sample = []
    # Validate initial_dataset dimensions
    num_datasets = len(initial_dataset)
    graphs_per_dataset = len(initial_dataset[0])

    # Ensure initial_dataset has enough datasets
    # required_datasets = (max(sample) // graphs_per_dataset) + 1
    # assert num_datasets >= required_datasets, f"initial_dataset must have at least {required_datasets} datasets."

    # Process the sample
    for i in sample:
        specific_dataset = i // graphs_per_dataset
        specific_graph = i % graphs_per_dataset
        training_sample.append(initial_dataset[specific_dataset][specific_graph])

    X_train = training_sample

    automl = AutoML(
        # Declare the input and output types
        input=XGraph,
        output=GraphSeqExplanation,
        objectives=(my_metric,),
        registry=[
            SubgraphXAutogoal],
        # Search space configuration
        search_timeout=300 * Hour,
        evaluation_timeout=24 * Hour,
        memory_limit=8 * Gb,
    )

    # Run the pipeline search process
    automl.fit(X_train, logger=[
        ConsoleLogger(),
        JsonLogger("experiment.json")
    ])

    # Report the best pipelines
    print(automl.best_pipelines_)
    print(automl.best_scores_)

# optimizer = subgraphX_autogoal_no_analyzer()
# masked_score, maskout_score, sparsity_score = optimizer.main()

# print(f"Masked score: {masked_score}")
# print(f"Maskout score: {maskout_score}")
# print(f"Sparsity score: {sparsity_score}")
if __name__=="__main__":
    run_autogoal()