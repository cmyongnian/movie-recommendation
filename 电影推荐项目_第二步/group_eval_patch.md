把 `src/group_eval.py` 做两处修改。

## 1）导入区增加 SVDPP
```python
from .baseline import ItemMeanModel
from .itemcf import ItemCF
from .mf import BiasMF
from .svdpp import SVDPP
from .gnn_feature import FeatureGNNRecommender
```

## 2）用下面这个完整函数替换原来的 `train_step3_models`
```python
def train_step3_models(
    ratings_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    itemcf_params: Dict,
    mf_params: Dict,
    svdpp_params: Dict,
    gnn_params: Dict,
):
    models = {}

    baseline_model = ItemMeanModel().fit(train_df)
    models["电影平均分"] = baseline_model

    itemcf_model = ItemCF(
        k=itemcf_params["k"],
        sim_metric=itemcf_params["sim_metric"],
        min_common=itemcf_params["min_common"],
    ).fit(train_df)
    models["基于物品的协同过滤"] = itemcf_model

    mf_model = BiasMF(
        n_factors=mf_params["n_factors"],
        lr=mf_params["lr"],
        reg=mf_params["reg"],
        epochs=mf_params["epochs"],
        seed=mf_params.get("seed", 42),
        verbose=False,
    ).fit(train_df, valid_df=valid_df)
    models["带偏置矩阵分解"] = mf_model

    svdpp_model = SVDPP(
        n_factors=svdpp_params["n_factors"],
        lr=svdpp_params["lr"],
        reg=svdpp_params["reg"],
        epochs=svdpp_params["epochs"],
        seed=svdpp_params.get("seed", 42),
        verbose=False,
    ).fit(train_df, valid_df=valid_df)
    models["SVD++"] = svdpp_model

    gnn_model = FeatureGNNRecommender(
        model_type=gnn_params["model_type"],
        hidden_dim=gnn_params["hidden_dim"],
        num_layers=gnn_params["num_layers"],
        lr=gnn_params["lr"],
        weight_decay=gnn_params["weight_decay"],
        epochs=gnn_params["epochs"],
        dropout=gnn_params.get("dropout", 0.1),
        seed=gnn_params.get("seed", 42),
        device=gnn_params.get("device", None),
        verbose=False,
    )
    gnn_model.fit(
        ratings_df=ratings_df,
        train_df=train_df,
        valid_df=valid_df,
        users_df=users_df,
        items_df=items_df,
    )
    models["图卷积网络"] = gnn_model

    return models
```
