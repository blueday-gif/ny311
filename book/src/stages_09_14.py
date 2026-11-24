"""Stages 9-14: time-split prediction, conformal sets, and a double-ML sketch.

Train on complaints created before 2026-08-01. Test on August-October 2026.
The label is whether the complaint was closed within 7 days.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from paths import INTERIM, RAW_CSV, REPORTS, FIGURES, SLIM_PQ, STORY_PQ, STORY_PQ_SQL, WEATHER  # noqa: F401

ROOT = Path(r"D:\nyc311")
PARQUET = SLIM_PQ
OUT = INTERIM / "stages_09_14.json"
CUTOFF = np.datetime64("2026-10-24T21:09:02")
TRAIN_END = np.datetime64("2026-08-01")
CAL_START = np.datetime64("2026-07-01")
SEED = 20261112


def load_frame() -> pd.DataFrame:
    con = duckdb.connect()
    frame = con.execute(
        f"""
        SELECT
          created,
          borough,
          agency,
          problem,
          channel,
          CASE
            WHEN closed IS NOT NULL AND closed < created THEN NULL
            WHEN closed IS NOT NULL AND closed >= created
              THEN date_diff('second', created, closed) <= 604800
            WHEN created <= TIMESTAMP '2026-10-24 21:09:02' THEN FALSE
            ELSE NULL
          END AS y
        FROM read_parquet('{PARQUET.as_posix()}')
        """
    ).fetchdf()
    labeled = frame.dropna(subset=["y", "borough", "agency", "problem", "channel"]).copy()
    labeled["y"] = labeled["y"].astype(bool)
    labeled["month"] = labeled["created"].dt.strftime("%Y-%m")
    return labeled


def pack(y_true: np.ndarray, prob: np.ndarray) -> dict:
    prob = np.clip(prob.astype(float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, prob >= 0.5)),
        "predicted_positive_rate": float((prob >= 0.5).mean()),
        "actual_positive_rate": float(np.mean(y_true)),
    }


def categorical_frame(frame: pd.DataFrame, columns: list[str], levels: dict[str, pd.Index]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for column in columns:
        out[column] = pd.Categorical(frame[column], categories=levels[column])
    return out


def fit_booster(train: pd.DataFrame, y: np.ndarray, n_estimators: int, max_depth: int, num_leaves: int) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=1.0 if n_estimators == 1 else 0.1,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=400,
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=-1,
        verbose=-1,
        random_state=SEED,
    )
    model.fit(train, np.asarray(y).astype(int), categorical_feature=list(train.columns))
    return model


def conformal(prob_cal: np.ndarray, y_cal: np.ndarray, prob_test: np.ndarray, y_test: np.ndarray) -> dict:
    alpha = 0.1
    p_true = np.where(y_cal, prob_cal, 1 - prob_cal)
    scores = np.sort(1 - p_true)
    n_cal = len(scores)
    rank = int(np.ceil((n_cal + 1) * (1 - alpha)))
    quantile = 1.0 if rank > n_cal else float(scores[rank - 1])
    threshold = 1 - quantile
    include_pos = prob_test >= threshold
    include_neg = (1 - prob_test) >= threshold
    covered = np.where(y_test, include_pos, include_neg)
    size = include_pos.astype(int) + include_neg.astype(int)
    return {
        "alpha": alpha,
        "target_coverage": 0.9,
        "calibration_n": int(n_cal),
        "score_quantile": quantile,
        "probability_threshold": float(threshold),
        "test_coverage": float(covered.mean()),
        "mean_set_size": float(size.mean()),
        "share_singleton": float(np.mean(size == 1)),
        "share_both": float(np.mean(size == 2)),
        "share_empty": float(np.mean(size == 0)),
    }


def double_ml(frame: pd.DataFrame) -> dict:
    subset = frame[frame["channel"].isin(["MOBILE", "PHONE"])].copy()
    subset["d"] = (subset["channel"] == "MOBILE").astype(int)
    y = subset["y"].to_numpy(dtype=float)
    d = subset["d"].to_numpy(dtype=float)
    naive = float(y[d == 1].mean() - y[d == 0].mean())
    levels = {column: pd.Index(sorted(subset[column].unique())) for column in ["borough", "agency", "problem", "month"]}
    features = categorical_frame(subset, ["borough", "agency", "problem", "month"], levels)
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, 2, size=len(subset))
    y_hat = np.zeros(len(subset))
    d_hat = np.zeros(len(subset))
    for held in (0, 1):
        train_idx = fold != held
        test_idx = fold == held
        outcome = fit_booster(features.iloc[train_idx], y[train_idx], 80, 6, 31)
        treatment = fit_booster(features.iloc[train_idx], d[train_idx], 80, 6, 31)
        y_hat[test_idx] = outcome.predict_proba(features.iloc[test_idx])[:, 1]
        d_hat[test_idx] = treatment.predict_proba(features.iloc[test_idx])[:, 1]
    y_res = y - y_hat
    d_res = d - d_hat
    denominator = float(np.mean(d_res**2))
    theta = float(np.mean(d_res * y_res) / denominator)
    influence = d_res * (y_res - theta * d_res) / denominator
    standard_error = float(np.std(influence, ddof=1) / np.sqrt(len(subset)))
    return {
        "contrast": "MOBILE minus PHONE",
        "outcome": "closed within 7 days",
        "controls": ["borough", "agency", "problem", "month"],
        "n": int(len(subset)),
        "mobile_n": int((d == 1).sum()),
        "phone_n": int((d == 0).sum()),
        "naive_difference": naive,
        "double_ml_difference": theta,
        "double_ml_se": standard_error,
        "double_ml_interval": [theta - 1.96 * standard_error, theta + 1.96 * standard_error],
    }


def main() -> None:
    print("loading")
    frame = load_frame()
    train = frame[frame["created"] < TRAIN_END]
    test = frame[frame["created"] >= TRAIN_END]
    columns = ["borough", "agency", "problem", "channel", "month"]
    levels = {column: pd.Index(sorted(train[column].unique())) for column in columns}
    x_train = categorical_frame(train, columns, levels)
    x_test = categorical_frame(test, columns, levels)
    y_train = train["y"].to_numpy()
    y_test = test["y"].to_numpy()
    print(f"train {len(train):,} test {len(test):,} train rate {y_train.mean():.3f} test rate {y_test.mean():.3f}")

    base_rate = float(y_train.mean())
    majority = pack(y_test, np.full(len(y_test), base_rate))

    print("single tree")
    tree = fit_booster(x_train, y_train, n_estimators=1, max_depth=4, num_leaves=16)
    tree_prob = tree.predict_proba(x_test)[:, 1]
    tree_metrics = pack(y_test, tree_prob)

    print("logistic")
    top_problems = train["problem"].value_counts().head(40).index
    logistic_frame = frame.copy()
    logistic_frame["problem_grouped"] = np.where(
        logistic_frame["problem"].isin(top_problems), logistic_frame["problem"], "그 밖"
    )
    logistic_columns = ["borough", "agency", "problem_grouped", "channel", "month"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    logistic_train = train.index
    logistic_test = test.index
    x_logistic_train = encoder.fit_transform(logistic_frame.loc[logistic_train, logistic_columns])
    x_logistic_test = encoder.transform(logistic_frame.loc[logistic_test, logistic_columns])
    logistic = LogisticRegression(max_iter=200, solver="saga", C=1.0, random_state=SEED)
    logistic.fit(x_logistic_train, y_train)
    logistic_prob = logistic.predict_proba(x_logistic_test)[:, 1]
    logistic_metrics = pack(y_test, logistic_prob)

    print("boosting")
    booster = fit_booster(x_train, y_train, n_estimators=120, max_depth=6, num_leaves=31)
    boost_prob = booster.predict_proba(x_test)[:, 1]
    boost_metrics = pack(y_test, boost_prob)

    print("conformal")
    cal_mask = (frame["created"] >= CAL_START) & (frame["created"] < TRAIN_END)
    early = frame[frame["created"] < CAL_START]
    calibration = frame[cal_mask]
    x_early = categorical_frame(early, columns, levels)
    x_cal = categorical_frame(calibration, columns, levels)
    conformal_model = fit_booster(x_early, early["y"].to_numpy(), 120, 6, 31)
    conformal_stats = conformal(
        conformal_model.predict_proba(x_cal)[:, 1],
        calibration["y"].to_numpy(),
        conformal_model.predict_proba(x_test)[:, 1],
        y_test,
    )

    print("mlp")
    rng = np.random.default_rng(SEED)
    sample_n = 150_000
    sample_pos = rng.choice(len(y_train), size=sample_n, replace=False)
    dense_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    dense_train = dense_encoder.fit_transform(logistic_frame.loc[logistic_train, logistic_columns].iloc[sample_pos])
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=25, early_stopping=True, random_state=SEED)
    mlp.fit(dense_train, y_train[sample_pos])
    mlp_prob = np.empty(len(y_test), dtype=float)
    test_features = logistic_frame.loc[logistic_test, logistic_columns]
    for start in range(0, len(y_test), 100_000):
        stop = min(start + 100_000, len(y_test))
        mlp_prob[start:stop] = mlp.predict_proba(dense_encoder.transform(test_features.iloc[start:stop]))[:, 1]
    mlp_metrics = pack(y_test, mlp_prob)

    print("double ml")
    dml = double_ml(frame)

    summary = {
        "label": "closed within 7 days of creation",
        "n_labeled": int(len(frame)),
        "train_end": "2026-08-01",
        "test_months": "2026-08 to 2026-10",
        "train_n": int(len(train)),
        "test_n": int(len(test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "censored_note": "Rows still open and created after 2026-10-24 21:09 are dropped because 7 days were not observed.",
        "models": {
            "majority": majority,
            "decision_tree": tree_metrics,
            "logistic": logistic_metrics,
            "lightgbm": boost_metrics,
            "mlp_sample_150k": mlp_metrics,
        },
        "mlp_train_rows": sample_n,
        "logistic_problem_levels": 41,
        "conformal": conformal_stats,
        "double_ml": dml,
        "dropped_unlabeled": int(3963959 - len(frame)),
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["models"], ensure_ascii=False, indent=2))
    print("conformal", json.dumps(conformal_stats))
    print("dml", json.dumps(dml))


if __name__ == "__main__":
    main()
