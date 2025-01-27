import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv, GatedGraphConv, GraphConv, GATv2Conv,
    RGCNConv, GINEConv, RGATConv, HEATConv, HANConv, HGTConv,
    global_mean_pool, global_max_pool, global_add_pool, LayerNorm, GCN, MLP
)
from arguments import arguments

args = arguments()

# Dictionary mapping for graph modules
MODULE_DICT = {
    'GCNConv': GCNConv,
    'GATConv': GATConv,
    'SAGEConv': SAGEConv,
    'GINConv': GINConv,
    'GatedGraphConv': GatedGraphConv,
    'GraphConv': GraphConv,
    'GATv2Conv': GATv2Conv
}

RMODULE_DICT = {
    'RGCNConv': RGCNConv,
    'RGATConv': RGATConv,
    'GINEConv': GINEConv,
    'HEATConv': HEATConv,
    'HANConv': HANConv,
    'HGTConv': HGTConv
}

def apply_pooling(x, batch, pooling_type):
    # Apply pooling to the graph embeddings based on the pooling type
    if pooling_type == 'mean':
        return global_mean_pool(x, batch)
    elif pooling_type == 'max':
        return global_max_pool(x, batch)
    elif pooling_type == 'sum':
        return global_add_pool(x, batch)
    else:
        raise ValueError(f"Unsupported pooling type: {pooling_type}")

class GNN_classifier(nn.Module):
    def __init__(self, size, num_layers, dropout, module, layer_norm=False, residual=False, pooling='mean', lin_transform=None, hidden_size=768):
        super(GNN_classifier, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual = residual
        self.pooling = pooling
        self.module = module
        self.size = size
        # Set hidden size based on module type (adjust for attention heads if needed)
        self.hidden_size = hidden_size if module in ['GATConv', 'GATv2Conv'] else size

        # Linear transformation for input features (optional)
        if lin_transform is not None:
            self.weight1 = nn.Parameter(torch.empty(lin_transform, size).to('cuda'))
            self.weight2 = nn.Parameter(torch.empty(lin_transform, size).to('cuda'))
            nn.init.xavier_uniform_(self.weight1)
            nn.init.xavier_uniform_(self.weight2)
            size = lin_transform

        # Define GNN layers
        self.gnn_layers1, self.layer_norms1 = self._build_gnn_layers(size, num_layers, module, layer_norm)
        self.gnn_layers2, self.layer_norms2 = self._build_gnn_layers(size, num_layers, module, layer_norm)

        # Residual layers for specific modules (only for GAT-based modules)
        if residual and module in ['GATConv', 'GATv2Conv']:
            self.residuals1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size * args['heads'])])
            self.residuals2 = nn.ModuleList([nn.Linear(hidden_size, hidden_size * args['heads'])])

        # Final MLP for classification after pooling both graph embeddings
        self.mlp = MLP(in_channels=size * 2, hidden_channels=self.hidden_size, out_channels=4, dropout=0.2, num_layers=4)

    def _build_gnn_layers(self, size, num_layers, module, layer_norm):
        # Helper method to build GNN layers and optional LayerNorm layers
        gnn_layers = nn.ModuleList()
        layer_norms = nn.ModuleList() if layer_norm else None

        for i in range(num_layers):
            if module in ['GCNConv', 'GraphConv', 'SAGEConv']:
                gnn_layers.append(MODULE_DICT[module](size, size))
            elif module in ['GATConv', 'GATv2Conv']:
                heads = args['heads']
                input_size = size if i == 0 else size * heads
                gnn_layers.append(MODULE_DICT[module](input_size, size, heads=heads))
            elif module == 'GINConv':
                gnn_layers.append(MODULE_DICT[module](nn=nn.Sequential(nn.Linear(size, size), nn.ReLU(), nn.Linear(size, size))))
            else:
                raise NotImplementedError(f"Unsupported module type: {module}")

            if layer_norm:
                layer_norms.append(LayerNorm(in_channels=size * (heads if module in ['GATConv', 'GATv2Conv'] else 1)))

        return gnn_layers, layer_norms

    def forward(self, x1, edge_index1, x2, edge_index2, batch1, batch2):
        # Apply optional linear transformation to inputs if specified
        if hasattr(self, 'weight1') and hasattr(self, 'weight2'):
            x1 = F.linear(x1, self.weight1)
            x2 = F.linear(x2, self.weight2)

        # Process each graph independently using GNN layers
        x1 = self._process_graph(x1, edge_index1, batch1, self.gnn_layers1, self.layer_norms1, self.residuals1 if self.residual else None)
        x2 = self._process_graph(x2, edge_index2, batch2, self.gnn_layers2, self.layer_norms2, self.residuals2 if self.residual else None)

        # Concatenate the embeddings from both graphs and apply MLP for classification
        x = torch.cat([x1, x2], dim=1)
        return self.mlp(x)

    def _process_graph(self, x, edge_index, batch, gnn_layers, layer_norms, residuals):
        # Helper method to process a graph through GNN layers
        for i, conv in enumerate(gnn_layers):
            if self.residual:
                prev_x = x.clone()  # Store the previous state for residual connection
            x = conv(x, edge_index)  # Apply GNN layer
            if self.residual:
                # Apply residual connection if enabled
                if residuals and i == 0:
                    x = x + residuals[0](prev_x)
                else:
                    x = x + prev_x
            x = F.relu(x)  # Apply ReLU activation
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout (except last layer)
                if self.layer_norm:
                    x = layer_norms[i](x, batch)  # Apply layer normalization if enabled
        return apply_pooling(x, batch, self.pooling)  # Apply pooling to obtain a single graph representation

class RGNN_classifier(nn.Module):
    def __init__(self, size, num_layers, dropout, module, num_relations, layer_norm=False, residual=False, pooling='mean', hidden_size=768):
        super(RGNN_classifier, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual = residual
        self.pooling = pooling
        self.module = module
        self.num_relations = num_relations
        # Set hidden size based on module type
        self.hidden_size = hidden_size if module == 'RGATConv' else size

        # Define GNN layers for heterogeneous graphs
        self.gnn_layers1, self.layer_norms1 = self._build_gnn_layers(size, num_layers, module, num_relations, layer_norm)
        self.gnn_layers2, self.layer_norms2 = self._build_gnn_layers(size, num_layers, module, num_relations, layer_norm)

        # Residual layers for specific modules (only for RGAT-based modules)
        if residual and module == 'RGATConv':
            self.residuals1 = nn.ModuleList([nn.Linear(hidden_size, hidden_size * args['heads'])])
            self.residuals2 = nn.ModuleList([nn.Linear(hidden_size, hidden_size * args['heads'])])

        # Final MLP for classification after pooling both graph embeddings
        self.mlp = MLP(size * 2, self.hidden_size, 4, 3, dropout)

    def _build_gnn_layers(self, size, num_layers, module, num_relations, layer_norm):
        # Helper method to build GNN layers and optional LayerNorm layers for heterogeneous graphs
        gnn_layers = nn.ModuleList()
        layer_norms = nn.ModuleList() if layer_norm else None

        for i in range(num_layers):
            if module == 'RGCNConv':
                gnn_layers.append(RMODULE_DICT[module](in_channels=size, out_channels=size, num_relations=num_relations))
            elif module == 'RGATConv':
                heads = args['heads'] if i > 0 else 1
                input_size = size * heads if i > 0 else size
                gnn_layers.append(RMODULE_DICT[module](in_channels=input_size, out_channels=size, heads=heads, num_relations=num_relations))
            elif module == 'GINEConv':
                gnn_layers.append(RMODULE_DICT[module](nn=nn.Sequential(nn.Linear(size, size), nn.ReLU(), nn.Linear(size, size))))
            else:
                raise NotImplementedError(f"Unsupported module type: {module}")

            if layer_norm:
                layer_norms.append(LayerNorm(in_channels=size * heads if module == 'RGATConv' else size))

        return gnn_layers, layer_norms

    def forward(self, x1, edge_index1, edge_type1, x2, edge_index2, edge_type2, batch1, batch2):
        # Process each graph independently using GNN layers
        x1 = self._process_graph(x1, edge_index1, edge_type1, batch1, self.gnn_layers1, self.layer_norms1, self.residuals1 if self.residual else None)
        x2 = self._process_graph(x2, edge_index2, edge_type2, batch2, self.gnn_layers2, self.layer_norms2, self.residuals2 if self.residual else None)

        # Concatenate the embeddings from both graphs and apply MLP for classification
        x = torch.cat([x1, x2], dim=1)
        return self.mlp(x)

    def _process_graph(self, x, edge_index, edge_type, batch, gnn_layers, layer_norms, residuals):
        # Helper method to process a heterogeneous graph through GNN layers
        for i, conv in enumerate(gnn_layers):
            if self.residual:
                prev_x = x.clone()  # Store the previous state for residual connection
            x = conv(x=x, edge_index=edge_index, edge_type=edge_type)  # Apply GNN layer with edge types
            if self.residual:
                # Apply residual connection if enabled
                if residuals and i == 0:
                    x = x + residuals[0](prev_x)
                else:
                    x = x + prev_x
            x = F.relu(x)  # Apply ReLU activation
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout (except last layer)
                if self.layer_norm:
                    x = layer_norms[i](x, batch)  # Apply layer normalization if enabled
        return apply_pooling(x, batch, self.pooling)  # Apply pooling to obtain a single graph representation

class GCN_original(GCN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout)
        # Define two GCNs and an MLP for classification
        self.GCN1 = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.GCN2 = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.mlp = MLP(in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout)

    def forward(self, x1, x2, edge_index1, edge_index2, batch1, batch2):
        # Process both input graphs independently through their respective GCNs
        x1 = self.GCN1(x1, edge_index1, batch=batch1)
        x2 = self.GCN2(x2, edge_index2, batch=batch2)
        # Pool the node embeddings from each graph to obtain a single representation
        x1 = global_max_pool(x1, batch1)
        x2 = global_max_pool(x2, batch2)
        # Concatenate the pooled embeddings and pass through the MLP for classification
        x = torch.cat([x1, x2], dim=1)
        return self.mlp(x)
