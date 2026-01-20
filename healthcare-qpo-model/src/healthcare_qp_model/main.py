from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


# =========================================================
# Synthetic healthcare-like data (NO PHI)
# =========================================================
def make_synthetic_data(seed: int = 7):
    """
    Creates:
      - enrollment table (member_id, enroll_start, enroll_end, age, is_commercial)
      - claims table (member_id, service_date, claim_type, allowed_amt, chronic_flag)

    The generator is designed to create enough "high-cost" members so the label
    isn't all zeros, and to behave reasonably across different cutoffs.
    """
    rng = np.random.default_rng(seed)

    n_members = 5000
    member_ids = np.arange(1, n_members + 1)

    # Enrollment: stagger starts + stagger ends
    enroll_start = pd.to_datetime("2017-01-01") + pd.to_timedelta(
        rng.integers(0, 365, n_members), unit="D"
    )
    enroll_end = pd.to_datetime("2019-12-31") - pd.to_timedelta(
        rng.integers(0, 365, n_members), unit="D"
    )
    enroll_end = np.maximum(
        enroll_end.values.astype("datetime64[ns]"),
        (enroll_start + pd.Timedelta(days=60)).values,
    )

    age = rng.integers(18, 65, n_members)
    is_commercial = (rng.random(n_members) < 0.85).astype(int)

    enrollment = pd.DataFrame(
        {
            "member_id": member_ids,
            "enroll_start": enroll_start,
            "enroll_end": pd.to_datetime(enroll_end),
            "age": age,
            "is_commercial": is_commercial,
        }
    )

    # Claims generation
    n_claims = 90000
    claim_dates = pd.to_datetime("2017-01-01") + pd.to_timedelta(
        rng.integers(
            0,
            (pd.to_datetime("2019-12-31") - pd.to_datetime("2017-01-01")).days + 1,
            n_claims,
        ),
        unit="D",
    )
    member_for_claim = rng.choice(member_ids, size=n_claims, replace=True)
    claim_type = rng.choice(["med", "rx"], size=n_claims, p=[0.72, 0.28])

    # Latent risk (heavy-tailed): makes a small fraction truly expensive
    member_risk = rng.lognormal(mean=0.0, sigma=0.9, size=n_members)

    # Cost base by type
    base_amt = np.where(
        claim_type == "med",
        rng.gamma(shape=2.2, scale=350.0, size=n_claims),  # med larger
        rng.gamma(shape=2.0, scale=120.0, size=n_claims),  # rx smaller
    )

    # Inflate costs by member risk and age
    age_factor = 1.0 + (age[member_for_claim - 1] - 18) / 120.0
    allowed_amt = base_amt * member_risk[member_for_claim - 1] * age_factor

    # A crude "chronic condition" flag (more likely for higher risk)
    chronic_prob = (0.05 * np.clip(member_risk[member_for_claim - 1], 0.0, 6.0)).clip(
        0.0, 0.5
    )
    chronic_flag = (rng.random(n_claims) < chronic_prob).astype(int)

    claims = pd.DataFrame(
        {
            "member_id": member_for_claim,
            "service_date": claim_dates,
            "claim_type": claim_type,
            "allowed_amt": allowed_amt,
            "chronic_flag": chronic_flag,
        }
    )

    # Keep only claims during enrollment (more realistic)
    claims = claims.merge(
        enrollment[["member_id", "enroll_start", "enroll_end"]],
        on="member_id",
        how="left",
    )
    claims = claims[
        (claims["service_date"] >= claims["enroll_start"])
        & (claims["service_date"] <= claims["enroll_end"])
    ].copy()
    claims.drop(columns=["enroll_start", "enroll_end"], inplace=True)

    return enrollment, claims


# =========================================================
# Cohort specification
# =========================================================
@dataclass(frozen=True)
class CohortSpec:
    """
    cutoff_date: the "freeze date" separating past vs future.
    lookback_days: history window length used for features.
    prediction_days: future window length used to build the target label.
    min_enrolled_days_before: QP requires enrollment for at least this many
                              days prior to the cutoff.
    """

    cutoff_date: str
    lookback_days: int = 365
    prediction_days: int = 180
    min_enrolled_days_before: int = 90


# =========================================================
# QP (Qualifying Population)
# =========================================================
def build_qp(enrollment: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
    """
    QP: who is eligible to be scored at the cutoff date.

    Example QP rules:
      - Commercial
      - age 18-64
      - enrolled on the cutoff date
      - enrolled for at least min_enrolled_days_before prior to cutoff
    """
    cutoff = pd.to_datetime(spec.cutoff_date)

    qp = enrollment[
        (enrollment["is_commercial"] == 1)
        & (enrollment["age"].between(18, 64))
        & (enrollment["enroll_start"] <= cutoff)
        & (enrollment["enroll_end"] >= cutoff)
        & (enrollment["enroll_start"] <= cutoff - pd.Timedelta(days=spec.min_enrolled_days_before))
    ][["member_id", "age"]].copy()

    return qp


# =========================================================
# Feature engineering (PAST only)
# =========================================================
def build_features(claims: pd.DataFrame, qp: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
    """
    Features use ONLY PAST claims:
      (cutoff - lookback_days, cutoff]  (service_date <= cutoff)

    We also use distinct service dates for visit/fill counts,
    matching the common 'distinct_srvc_dates = 1' requirement.
    """
    cutoff = pd.to_datetime(spec.cutoff_date)
    start = cutoff - pd.Timedelta(days=spec.lookback_days)

    past = claims[(claims["service_date"] > start) & (claims["service_date"] <= cutoff)].copy()
    past = past.merge(qp[["member_id"]], on="member_id", how="inner")
    past["svc_day"] = past["service_date"].dt.floor("D")

    med = past[past["claim_type"] == "med"]
    rx = past[past["claim_type"] == "rx"]

    med_agg = med.groupby("member_id").agg(
        past_med_allowed=("allowed_amt", "sum"),
        past_med_visits=("svc_day", "nunique"),
        chronic_claims=("chronic_flag", "sum"),
    )

    rx_agg = rx.groupby("member_id").agg(
        past_rx_allowed=("allowed_amt", "sum"),
        past_rx_fills=("svc_day", "nunique"),
    )

    feats = (
        qp.merge(med_agg, on="member_id", how="left")
        .merge(rx_agg, on="member_id", how="left")
        .fillna(
            {
                "past_med_allowed": 0.0,
                "past_med_visits": 0.0,
                "chronic_claims": 0.0,
                "past_rx_allowed": 0.0,
                "past_rx_fills": 0.0,
            }
        )
    )

    feats["past_total_allowed"] = feats["past_med_allowed"] + feats["past_rx_allowed"]
    feats["past_total_events"] = feats["past_med_visits"] + feats["past_rx_fills"]

    return feats


# =========================================================
# Target population (FUTURE only)
# =========================================================
def build_target_population(
    claims: pd.DataFrame,
    qp: pd.DataFrame,
    spec: CohortSpec,
    *,
    threshold: float | None = None,
    top_pct: float = 0.10,
) -> pd.DataFrame:
    """
    Target population = QP members + outcome defined in FUTURE window:
      (cutoff, cutoff + prediction_days]

    We offer two target definitions:
      A) Relative "high cost" (default): top `top_pct` of future cost among QP
      B) Absolute "high cost": future_cost >= threshold

    Relative targets are common in risk stratification and avoid "one class only"
    failures in small or synthetic datasets.
    """
    cutoff = pd.to_datetime(spec.cutoff_date)
    end = cutoff + pd.Timedelta(days=spec.prediction_days)

    future = claims[(claims["service_date"] > cutoff) & (claims["service_date"] <= end)].copy()
    future = future.merge(qp[["member_id"]], on="member_id", how="inner")

    fut_agg = future.groupby("member_id", as_index=False).agg(
        future_allowed=("allowed_amt", "sum")
    )

    target = qp[["member_id"]].merge(fut_agg, on="member_id", how="left")
    target["future_allowed"] = target["future_allowed"].fillna(0.0)

    if threshold is None:
        # percentile threshold among QP members
        cutoff_value = float(target["future_allowed"].quantile(1.0 - top_pct))
        target["y_high_cost"] = (target["future_allowed"] >= cutoff_value).astype(int)
        target["label_rule"] = f"top_{int(top_pct*100)}pct_future_cost (>= {cutoff_value:,.2f})"
    else:
        target["y_high_cost"] = (target["future_allowed"] >= float(threshold)).astype(int)
        target["label_rule"] = f"future_cost >= {float(threshold):,.2f}"

    return target


# =========================================================
# Train + OOT validate
# =========================================================
def run_example() -> int:
    enrollment, claims = make_synthetic_data(seed=7)

    # Training and Out-Of-Time cutoff dates (as in your screenshot concept)
    train_spec = CohortSpec(cutoff_date="2018-06-30", lookback_days=365, prediction_days=180)
    oot_spec = CohortSpec(cutoff_date="2018-12-31", lookback_days=365, prediction_days=180)

    # QP
    qp_train = build_qp(enrollment, train_spec)
    qp_oot = build_qp(enrollment, oot_spec)

    # Features (past only)
    X_train = build_features(claims, qp_train, train_spec)
    X_oot = build_features(claims, qp_oot, oot_spec)

    # Target population (future only)
    # Default: label top 10% by future cost (ensures both classes exist)
    y_train = build_target_population(claims, qp_train, train_spec, threshold=None, top_pct=0.10)
    y_oot = build_target_population(claims, qp_oot, oot_spec, threshold=None, top_pct=0.10)

    # Build final model driver datasets
    train = X_train.merge(y_train[["member_id", "future_allowed", "y_high_cost", "label_rule"]], on="member_id", how="inner")
    oot = X_oot.merge(y_oot[["member_id", "future_allowed", "y_high_cost", "label_rule"]], on="member_id", how="inner")

    # Feature columns
    features = [
        "age",
        "past_med_allowed",
        "past_rx_allowed",
        "past_med_visits",
        "past_rx_fills",
        "chronic_claims",
        "past_total_allowed",
        "past_total_events",
    ]

    # Quick class distribution sanity checks
    train_counts = train["y_high_cost"].value_counts().to_dict()
    oot_counts = oot["y_high_cost"].value_counts().to_dict()

    print("\n=== Cohort Summary ===")
    print(f"Training cutoff: {train_spec.cutoff_date} | QP size: {len(qp_train):,} | Label: {train['label_rule'].iloc[0]}")
    print(f"OOT cutoff:      {oot_spec.cutoff_date} | QP size: {len(qp_oot):,} | Label: {oot['label_rule'].iloc[0]}")
    print(f"Training class counts: {train_counts}")
    print(f"OOT class counts:      {oot_counts}")

    if train["y_high_cost"].nunique() < 2:
        raise RuntimeError(
            "Training target has only one class; adjust top_pct/threshold/prediction_days "
            "or synthetic generator severity."
        )

    # Train model
    model = LogisticRegression(max_iter=300)
    model.fit(train[features], train["y_high_cost"])

    # Evaluate
    p_train = model.predict_proba(train[features])[:, 1]
    p_oot = model.predict_proba(oot[features])[:, 1]

    train_auc = roc_auc_score(train["y_high_cost"], p_train)
    oot_auc = roc_auc_score(oot["y_high_cost"], p_oot)

    train_ap = average_precision_score(train["y_high_cost"], p_train)
    oot_ap = average_precision_score(oot["y_high_cost"], p_oot)

    print("\n=== Performance ===")
    print(f"Train AUC: {train_auc:.3f} | Train AP: {train_ap:.3f}")
    print(f"OOT   AUC: {oot_auc:.3f} | OOT   AP: {oot_ap:.3f}")

    # Show a few highest future-cost members in OOT
    show = oot[["member_id", "age", "past_total_allowed", "future_allowed", "y_high_cost"]].copy()
    show = show.sort_values("future_allowed", ascending=False).head(10)

    print("\n=== Top 10 OOT members by future cost (label window) ===")
    print(show.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(run_example())
