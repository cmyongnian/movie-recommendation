import json
from pathlib import Path

import pandas as pd

from .baseline import GlobalMeanModel, UserMeanModel, ItemMeanModel
from .itemcf import ItemCF
from .mf import BiasMF
from .svdpp import SVDPP
from .metrics import evaluate_model


def _build_result_row(prefix: str, metrics: dict) -> dict:
    return {
        f"{prefix}_mae": metrics["mae"],
        f"{prefix}_rmse": metrics["rmse"],
        f"{prefix}_exact_acc": metrics["exact_acc"],
        f"{prefix}_within_0_5_acc": metrics["within_0_5_acc"],
        f"{prefix}_within_1_0_acc": metrics["within_1_0_acc"],
        f"{prefix}_like_acc": metrics["like_acc"],
        f"{prefix}_precision": metrics["precision"],
        f"{prefix}_recall": metrics["recall"],
        f"{prefix}_f1": metrics["f1"],
    }


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

        row = {"model": model.name}
        row.update(_build_result_row("valid", valid_metrics))
        row.update(_build_result_row("test", test_metrics))
        results.append(row)

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
            }
            row.update(_build_result_row("valid", valid_metrics))
            row.update(_build_result_row("test", test_metrics))
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
                }
                row.update(_build_result_row("valid", valid_metrics))
                row.update(_build_result_row("test", test_metrics))

                if hasattr(model, "best_epoch") and model.best_epoch is not None:
                    row["best_epoch"] = int(model.best_epoch)
                if hasattr(model, "best_valid_rmse") and model.best_valid_rmse is not None:
                    row["tracked_best_valid_rmse"] = round(float(model.best_valid_rmse), 6)
                if hasattr(model, "early_stopped"):
                    row["early_stopped"] = bool(model.early_stopped)

                results.append(row)

                if valid_metrics["rmse"] < best_valid_rmse:
                    best_valid_rmse = valid_metrics["rmse"]
                    best_model = model

    result_df = pd.DataFrame(results).sort_values("valid_rmse").reset_index(drop=True)
    return result_df, best_model


def sweep_svdpp(
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
                model = SVDPP(
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
                    "model": "SVDPP",
                    "n_factors": n_factors,
                    "lr": lr,
                    "reg": reg,
                    "epochs": epochs,
                }
                row.update(_build_result_row("valid", valid_metrics))
                row.update(_build_result_row("test", test_metrics))

                if hasattr(model, "best_epoch") and model.best_epoch is not None:
                    row["best_epoch"] = int(model.best_epoch)
                if hasattr(model, "best_valid_rmse") and model.best_valid_rmse is not None:
                    row["tracked_best_valid_rmse"] = round(float(model.best_valid_rmse), 6)
                if hasattr(model, "early_stopped"):
                    row["early_stopped"] = bool(model.early_stopped)

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
    svdpp_results: pd.DataFrame | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_results.to_csv(output_dir / "baseline_results.csv", index=False, encoding="utf-8-sig")
    itemcf_results.to_csv(output_dir / "itemcf_results.csv", index=False, encoding="utf-8-sig")
    mf_results.to_csv(output_dir / "mf_results.csv", index=False, encoding="utf-8-sig")
    if svdpp_results is not None and len(svdpp_results) > 0:
        svdpp_results.to_csv(output_dir / "svdpp_results.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)