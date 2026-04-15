from copy import deepcopy
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_utils import (
    build_graph_data,
    dataframe_to_index_tensors,
    load_step1_feature_tables,
)
from .metrics import mae_rmse, clip_rating


def _build_sparse_operators(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device,
):
    row = edge_index[0]
    col = edge_index[1]

    self_loop = torch.arange(num_nodes, device=device)
    row = torch.cat([row, self_loop], dim=0)
    col = torch.cat([col, self_loop], dim=0)

    self_loop_values = torch.ones(num_nodes, dtype=torch.float32, device=device)
    values = torch.cat([edge_weight, self_loop_values], dim=0)

    adj = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=values,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()

    idx = adj.indices()
    val = adj.values()

    row = idx[0]
    col = idx[1]

    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.index_add_(0, row, val)
    degree = degree.clamp(min=1.0)

    deg_inv_sqrt = degree.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    norm_values = val * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    mean_values = val / degree[row]

    adj_norm = torch.sparse_coo_tensor(
        indices=idx,
        values=norm_values,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()

    adj_mean = torch.sparse_coo_tensor(
        indices=idx,
        values=mean_values,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()

    return adj_norm, adj_mean


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.sparse.mm(adj_norm, x)
        return x


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, adj_mean: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(adj_mean, x)
        out = torch.cat([x, neigh], dim=1)
        out = self.linear(out)
        return out


class FeatureGNNModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        num_users: int,
        num_items: int,
        user_input_dim: int,
        item_input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        feature_dropout: float = 0.1,
    ):
        super().__init__()

        if model_type not in {"gcn", "graphsage"}:
            raise ValueError("model_type 只能是 'gcn' 或 'graphsage'")

        self.model_type = model_type
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.feature_dropout = feature_dropout

        # 属性特征编码
        self.user_encoder = nn.Sequential(
            nn.Linear(user_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.item_encoder = nn.Sequential(
            nn.Linear(item_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # ID embedding
        self.user_id_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_id_embedding = nn.Embedding(num_items, hidden_dim)

        # 属性 + ID 融合
        self.user_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.item_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        layers = []
        for _ in range(num_layers):
            if model_type == "gcn":
                layers.append(GCNLayer(hidden_dim, hidden_dim))
            else:
                layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        # 偏置项
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # MLP 评分头
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.register_buffer("global_mean", torch.tensor(0.0, dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def encode_nodes(self, graph_data: Dict) -> torch.Tensor:
        user_feat = graph_data["user_features"]
        item_feat = graph_data["item_features"]

        user_feat = self.user_encoder(user_feat)
        item_feat = self.item_encoder(item_feat)

        user_ids = torch.arange(self.num_users, device=user_feat.device)
        item_ids = torch.arange(self.num_items, device=item_feat.device)

        user_id_emb = self.user_id_embedding(user_ids)
        item_id_emb = self.item_id_embedding(item_ids)

        user_x = self.user_fusion(torch.cat([user_feat, user_id_emb], dim=1))
        item_x = self.item_fusion(torch.cat([item_feat, item_id_emb], dim=1))

        user_x = F.dropout(user_x, p=self.feature_dropout, training=self.training)
        item_x = F.dropout(item_x, p=self.feature_dropout, training=self.training)

        x0 = torch.cat([user_x, item_x], dim=0)

        outputs = [x0]
        x = x0

        for layer_idx, layer in enumerate(self.layers):
            if self.model_type == "gcn":
                x = layer(x, graph_data["adj_norm"])
            else:
                x = layer(x, graph_data["adj_mean"])

            if layer_idx != len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            outputs.append(x)

        # 多层表示平均，减少过平滑影响
        final_x = torch.mean(torch.stack(outputs, dim=0), dim=0)
        return final_x

    def predict_raw_index(self, user_idx: torch.Tensor, item_idx: torch.Tensor, graph_data: Dict) -> torch.Tensor:
        z = self.encode_nodes(graph_data)

        z_user = z[user_idx]
        z_item = z[item_idx + self.num_users]

        pair_feature = torch.cat(
            [
                z_user,
                z_item,
                z_user * z_item,
                torch.abs(z_user - z_item),
            ],
            dim=1,
        )

        interaction_score = self.scorer(pair_feature).squeeze(-1)

        pred = (
            self.global_mean
            + self.user_bias(user_idx).squeeze(-1)
            + self.item_bias(item_idx).squeeze(-1)
            + interaction_score
        )
        return pred

    def predict_index(self, user_idx: torch.Tensor, item_idx: torch.Tensor, graph_data: Dict) -> torch.Tensor:
        pred = self.predict_raw_index(user_idx, item_idx, graph_data)
        pred = pred.clamp(1.0, 5.0)
        return pred


class FeatureGNNRecommender:
    def __init__(
        self,
        model_type: str = "gcn",
        hidden_dim: int = 64,
        num_layers: int = 1,
        lr: float = 0.005,
        weight_decay: float = 1e-5,
        epochs: int = 30,
        dropout: float = 0.1,
        feature_dropout: float = 0.1,
        grad_clip: float = 5.0,
        patience: int = 8,
        seed: int = 42,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.grad_clip = grad_clip
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.name = (
            f"{model_type.upper()}("
            f"hidden={hidden_dim}, layers={num_layers}, "
            f"lr={lr}, wd={weight_decay}, epochs={epochs})"
        )

        self.graph_data: Optional[Dict] = None
        self.model: Optional[FeatureGNNModel] = None
        self.history = []

    def fit(
        self,
        ratings_df: pd.DataFrame,
        train_df: pd.DataFrame,
        valid_df: Optional[pd.DataFrame],
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
    ):
        torch.manual_seed(self.seed)

        self.graph_data = build_graph_data(
            ratings_df=ratings_df,
            train_df=train_df,
            users_df=users_df,
            items_df=items_df,
            device=self.device,
        )

        self.graph_data["adj_norm"], self.graph_data["adj_mean"] = _build_sparse_operators(
            num_nodes=self.graph_data["num_nodes"],
            edge_index=self.graph_data["edge_index"],
            edge_weight=self.graph_data["edge_weight"],
            device=self.device,
        )

        self.model = FeatureGNNModel(
            model_type=self.model_type,
            num_users=self.graph_data["num_users"],
            num_items=self.graph_data["num_items"],
            user_input_dim=self.graph_data["user_features"].shape[1],
            item_input_dim=self.graph_data["item_features"].shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            feature_dropout=self.feature_dropout,
        ).to(self.device)

        self.model.global_mean.fill_(float(train_df["rating"].mean()))

        train_user_idx, train_item_idx, train_ratings = dataframe_to_index_tensors(
            train_df,
            self.graph_data["user_id_to_idx"],
            self.graph_data["item_id_to_idx"],
            device=self.device,
        )

        valid_user_idx = valid_item_idx = valid_ratings = None
        if valid_df is not None and len(valid_df) > 0:
            valid_user_idx, valid_item_idx, valid_ratings = dataframe_to_index_tensors(
                valid_df,
                self.graph_data["user_id_to_idx"],
                self.graph_data["item_id_to_idx"],
                device=self.device,
            )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
        )

        best_state = None
        best_valid_rmse = float("inf")
        wait = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            pred = self.model.predict_raw_index(train_user_idx, train_item_idx, self.graph_data)
            mse_loss = F.mse_loss(pred, train_ratings)

            # 轻微 embedding 正则
            reg_loss = 0.0
            for name, param in self.model.named_parameters():
                if "embedding" in name and param.requires_grad:
                    reg_loss = reg_loss + 1e-6 * torch.sum(param * param)

            loss = mse_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()

            record = {
                "epoch": epoch,
                "train_loss": round(float(loss.item()), 6),
                "train_mse": round(float(mse_loss.item()), 6),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            }

            if valid_user_idx is not None and len(valid_user_idx) > 0:
                valid_metrics = self._evaluate_by_index(valid_user_idx, valid_item_idx, valid_ratings)
                record["valid_mae"] = valid_metrics["mae"]
                record["valid_rmse"] = valid_metrics["rmse"]

                scheduler.step(valid_metrics["rmse"])

                if valid_metrics["rmse"] < best_valid_rmse:
                    best_valid_rmse = valid_metrics["rmse"]
                    best_state = deepcopy(self.model.state_dict())
                    wait = 0
                else:
                    wait += 1

                if wait >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    self.history.append(record)
                    break
            else:
                scheduler.step(float(loss.item()))

            self.history.append(record)

            if self.verbose:
                print(record)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def _evaluate_by_index(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        ratings: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            pred = self.model.predict_index(user_idx, item_idx, self.graph_data)

        return mae_rmse(
            ratings.detach().cpu().numpy().tolist(),
            pred.detach().cpu().numpy().tolist(),
        )

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        user_idx, item_idx, ratings = dataframe_to_index_tensors(
            df,
            self.graph_data["user_id_to_idx"],
            self.graph_data["item_id_to_idx"],
            device=self.device,
        )
        return self._evaluate_by_index(user_idx, item_idx, ratings)

    def predict(self, user_id: int, item_id: int) -> float:
        if self.graph_data is None or self.model is None:
            raise RuntimeError("模型尚未训练")

        if user_id not in self.graph_data["user_id_to_idx"]:
            return clip_rating(float(self.model.global_mean.item()))
        if item_id not in self.graph_data["item_id_to_idx"]:
            return clip_rating(float(self.model.global_mean.item()))

        user_idx = torch.tensor(
            [self.graph_data["user_id_to_idx"][user_id]],
            dtype=torch.long,
            device=self.device,
        )
        item_idx = torch.tensor(
            [self.graph_data["item_id_to_idx"][item_id]],
            dtype=torch.long,
            device=self.device,
        )

        self.model.eval()
        with torch.no_grad():
            pred = self.model.predict_index(user_idx, item_idx, self.graph_data)[0].item()

        return clip_rating(float(pred))


def sweep_gnn_feature(
    ratings_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    users_path,
    items_path,
    model_types: list[str],
    hidden_dims: list[int],
    num_layers_list: list[int],
    lr_list: list[float],
    weight_decay_list: list[float],
    epochs: int = 30,
    dropout: float = 0.1,
    seed: int = 42,
    device: Optional[str] = None,
):
    users_df, items_df = load_step1_feature_tables(users_path, items_path)

    results = []
    best_model = None
    best_valid_rmse = float("inf")

    for model_type in model_types:
        for hidden_dim in hidden_dims:
            for num_layers in num_layers_list:
                for lr in lr_list:
                    for weight_decay in weight_decay_list:
                        model = FeatureGNNRecommender(
                            model_type=model_type,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            lr=lr,
                            weight_decay=weight_decay,
                            epochs=epochs,
                            dropout=dropout,
                            feature_dropout=dropout,
                            seed=seed,
                            device=device,
                            verbose=False,
                        )

                        model.fit(
                            ratings_df=ratings_df,
                            train_df=train_df,
                            valid_df=valid_df,
                            users_df=users_df,
                            items_df=items_df,
                        )

                        valid_metrics = model.evaluate(valid_df)
                        test_metrics = model.evaluate(test_df)

                        row = {
                            "model": model_type.upper() if model_type == "gcn" else "GraphSAGE",
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "epochs": epochs,
                            "valid_mae": valid_metrics["mae"],
                            "valid_rmse": valid_metrics["rmse"],
                            "test_mae": test_metrics["mae"],
                            "test_rmse": test_metrics["rmse"],
                        }
                        results.append(row)

                        if valid_metrics["rmse"] < best_valid_rmse:
                            best_valid_rmse = valid_metrics["rmse"]
                            best_model = model

    result_df = pd.DataFrame(results).sort_values("valid_rmse").reset_index(drop=True)
    return result_df, best_model