from pathlib import Path
import json

import pandas as pd

from .baseline import GlobalMeanModel, UserMeanModel, ItemMeanModel
from .itemcf import ItemCF
from .mf import BiasMF
from .metrics import evaluate_model


def run_baselines(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    models = [
        GlobalMeanModel(),
        UserMeanModel(),
        ItemMeanModel(),
    ]

    results = []
    for model in models:
        model.fit(train_df)

        valid_metrics = evaluate_model(model, valid_df)
        test_metrics = evaluate_model(model, test_df)

        results.append(
            {
                "model": model.name,
                "valid_mae": valid_metrics["mae"],
                "valid_rmse": valid_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            }
        )

    return pd.DataFrame(results).sort_values("valid_rmse").reset_index(drop=True)


def sweep_itemcf(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k_list: list[int],
    sim_metrics: list[str],
    min_common: int = 2,
):
    results = []
    best_model = None
    best_valid_rmse = float("inf")

    for sim_metric in sim_metrics:
        for k in k_list:
            model = ItemCF(k=k, sim_metric=sim_metric, min_common=min_common)
            model.fit(train_df)

            valid_metrics = evaluate_model(model, valid_df)
            test_metrics = evaluate_model(model, test_df)

            row = {
                "model": "ItemCF",
                "k": k,
                "sim_metric": sim_metric,
                "min_common": min_common,
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


def sweep_mf(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_factors_list: list[int],
    lr_list: list[float],
    reg_list: list[float],
    epochs: int = 20,
    seed: int = 42,
):
    results = []
    best_model = None
    best_valid_rmse = float("inf")

    for n_factors in n_factors_list:
        for lr in lr_list:
            for reg in reg_list:
                model = BiasMF(
                    n_factors=n_factors,
                    lr=lr,
                    reg=reg,
                    epochs=epochs,
                    seed=seed,
                    verbose=False,
                )
                model.fit(train_df, valid_df=valid_df)

                valid_metrics = evaluate_model(model, valid_df)
                test_metrics = evaluate_model(model, test_df)

                row = {
                    "model": "BiasMF",
                    "n_factors": n_factors,
                    "lr": lr,
                    "reg": reg,
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


def save_experiment_outputs(
    output_dir: Path,
    baseline_results: pd.DataFrame,
    itemcf_results: pd.DataFrame,
    mf_results: pd.DataFrame,
    summary: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_results.to_csv(output_dir / "baseline_results.csv", index=False, encoding="utf-8-sig")
    itemcf_results.to_csv(output_dir / "itemcf_results.csv", index=False, encoding="utf-8-sig")
    mf_results.to_csv(output_dir / "mf_results.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)