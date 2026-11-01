
from decimal import Decimal, InvalidOperation

from django.db import transaction

from scheduler.models import (
    WorkloadRequest,
    SchedulingDecision,
    CandidateEvaluation,
)

from ml_engine.prediction_service import (
    predict_historical,
    predict_persistence,
)


# ============================================================
# PROVIDER CONFIGURATION
# ============================================================

PROVIDER_ZONES = {
    "AWS": "IE",
    "AZURE": "DE",
    "GCP": "FR",
}

DEFAULT_WEIGHTS = {
    "carbon": Decimal("0.40"),
    "cost": Decimal("0.25"),
    "latency": Decimal("0.20"),
    "resources": Decimal("0.15"),
}


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_weights(weights):
    """Validate scoring weights and convert them to Decimal."""

    if not isinstance(weights, dict):
        raise ValueError("Weights must be supplied as a dictionary.")

    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(
            "Weights must contain carbon, cost, latency and resources."
        )

    try:
        converted = {
            key: Decimal(str(value))
            for key, value in weights.items()
        }
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("All scoring weights must be valid numbers.")

    if any(not value.is_finite() for value in converted.values()):
        raise ValueError("Scoring weights must be finite numbers.")

    if any(value < 0 for value in converted.values()):
        raise ValueError("Scoring weights cannot be negative.")

    if sum(converted.values()) != Decimal("1.00"):
        raise ValueError("Scoring weights must sum to exactly 1.00.")

    return converted


def validate_candidate(candidate):
    """Validate candidate metrics and workload resource requirements."""

    required_fields = [
        "provider",
        "carbon_intensity",
        "estimated_cost_per_hour",
        "estimated_latency_ms",
        "available_cpu_millicores",
        "available_memory_mb",
        "required_cpu_millicores",
        "required_memory_mb",
    ]

    missing_fields = [
        field
        for field in required_fields
        if candidate.get(field) is None
    ]

    if missing_fields:
        raise ValueError(
            f"Candidate is missing required fields: {missing_fields}"
        )

    if candidate["provider"] not in PROVIDER_ZONES:
        raise ValueError(
            f"Unsupported provider: {candidate['provider']}"
        )

    for field in (
        "required_cpu_millicores",
        "required_memory_mb",
        "available_cpu_millicores",
        "available_memory_mb",
    ):
        value = candidate[field]

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer.")

    if candidate["required_cpu_millicores"] <= 0:
        raise ValueError("Required CPU must be greater than zero.")

    if candidate["required_memory_mb"] <= 0:
        raise ValueError("Required memory must be greater than zero.")

    for field in (
        "carbon_intensity",
        "estimated_cost_per_hour",
        "estimated_latency_ms",
    ):
        try:
            value = Decimal(str(candidate[field]))
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError(f"{field} must be a valid number.")

        if not value.is_finite() or value < 0:
            raise ValueError(
                f"{field} must be finite and non-negative."
            )


# ============================================================
# CANDIDATE ELIGIBILITY
# ============================================================

def evaluate_eligibility(workload, candidate):
    """Check whether a provider satisfies workload constraints."""

    reasons = []

    required_fields = [
        "carbon_intensity",
        "estimated_cost_per_hour",
        "estimated_latency_ms",
        "available_cpu_millicores",
        "available_memory_mb",
    ]

    for field in required_fields:
        if candidate.get(field) is None:
            reasons.append(f"Missing required metric: {field}")

    if reasons:
        return False, reasons

    for field in (
        "carbon_intensity",
        "estimated_cost_per_hour",
        "estimated_latency_ms",
    ):
        try:
            value = Decimal(str(candidate[field]))
        except (InvalidOperation, TypeError, ValueError):
            reasons.append(f"Invalid numeric metric: {field}")
            continue

        if not value.is_finite() or value < 0:
            reasons.append(
                f"Invalid {field.replace('_', ' ')}."
            )

    if reasons:
        return False, reasons

    if candidate["available_cpu_millicores"] < workload.cpu_millicores:
        reasons.append("Insufficient CPU capacity.")

    if candidate["available_memory_mb"] < workload.memory_mb:
        reasons.append("Insufficient memory capacity.")

    if (
        workload.max_latency_ms is not None
        and Decimal(str(candidate["estimated_latency_ms"]))
        > Decimal(str(workload.max_latency_ms))
    ):
        reasons.append("Latency exceeds workload limit.")

    if (
        workload.max_cost_per_hour is not None
        and Decimal(str(candidate["estimated_cost_per_hour"]))
        > Decimal(str(workload.max_cost_per_hour))
    ):
        reasons.append("Cost exceeds workload limit.")

    return len(reasons) == 0, reasons


# ============================================================
# OBJECTIVE NORMALISATION
# ============================================================

def normalise_objective(values, lower_is_better=True):
    """Normalise objective values to a 0-1 scale."""

    if not values:
        return []

    values = [Decimal(str(value)) for value in values]

    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return [Decimal("0.5") for _ in values]

    normalised = []

    for value in values:
        score = (value - minimum) / (maximum - minimum)

        if not lower_is_better:
            score = Decimal("1") - score

        normalised.append(score)

    return normalised


# ============================================================
# MULTI-OBJECTIVE SCORING
# ============================================================

def calculate_candidate_scores(candidates, weights):
    """Calculate weighted scores for eligible candidates."""

    if not candidates:
        return []

    weights = validate_weights(weights)

    for candidate in candidates:
        validate_candidate(candidate)

    carbon_scores = normalise_objective(
        [c["carbon_intensity"] for c in candidates]
    )

    cost_scores = normalise_objective(
        [c["estimated_cost_per_hour"] for c in candidates]
    )

    latency_scores = normalise_objective(
        [c["estimated_latency_ms"] for c in candidates]
    )

    resource_values = [
        min(
            Decimal(str(c["available_cpu_millicores"]))
            / Decimal(str(c["required_cpu_millicores"])),
            Decimal(str(c["available_memory_mb"]))
            / Decimal(str(c["required_memory_mb"])),
        )
        for c in candidates
    ]

    resource_scores = normalise_objective(
        resource_values,
        lower_is_better=False,
    )

    scored_candidates = []

    for index, candidate in enumerate(candidates):
        score = (
            weights["carbon"] * carbon_scores[index]
            + weights["cost"] * cost_scores[index]
            + weights["latency"] * latency_scores[index]
            + weights["resources"] * resource_scores[index]
        )

        scored_candidates.append({
            **candidate,
            "objective_score": score,
        })

    return sorted(
        scored_candidates,
        key=lambda item: (
            item["objective_score"],
            item["provider"],
        ),
    )


# ============================================================
# ML-BASED PROVIDER CANDIDATE GENERATION
# ============================================================

def build_predicted_candidates(
    workload,
    prediction_time,
    provider_metrics,
    prediction_method="random_forest",
):
    """
    Build provider candidates using a configurable carbon
    forecasting method.

    Supported prediction methods:
        - random_forest
        - persistence

    Workload resource requirements are obtained directly
    from the WorkloadRequest model instance.

    Provider cost, latency and capacity metrics must be
    supplied separately by the experiment.
    """

    prediction_functions = {
        "random_forest": predict_historical,
        "persistence": predict_persistence,
    }

    if prediction_method not in prediction_functions:
        raise ValueError(
            f"Unsupported prediction method: {prediction_method}"
        )

    if not isinstance(provider_metrics, dict):
        raise ValueError("Provider metrics must be a dictionary.")

    if set(provider_metrics) != set(PROVIDER_ZONES):
        raise ValueError(
            "Provider metrics must contain exactly AWS, AZURE and GCP."
        )

    required_metric_fields = [
        "estimated_cost_per_hour",
        "estimated_latency_ms",
        "available_cpu_millicores",
        "available_memory_mb",
    ]

    candidates = []

    prediction_function = prediction_functions[prediction_method]

    for provider, zone in PROVIDER_ZONES.items():

        metrics = provider_metrics[provider]

        if not isinstance(metrics, dict):
            raise ValueError(
                f"Metrics for {provider} must be a dictionary."
            )

        missing_fields = [
            field
            for field in required_metric_fields
            if metrics.get(field) is None
        ]

        if missing_fields:
            raise ValueError(
                f"{provider} is missing metrics: {missing_fields}"
            )

        prediction = prediction_function(
            zone=zone,
            prediction_time=prediction_time,
        )

        if prediction["zone"] != zone:
            raise ValueError(
                f"Prediction zone mismatch for {provider}."
            )

        if prediction["forecast_horizon_hours"] != 1:
            raise ValueError(
                f"Unexpected forecast horizon for {provider}."
            )

        candidates.append({
            "provider": provider,

            "carbon_intensity": Decimal(
                str(prediction["predicted_carbon_intensity"])
            ),

            "estimated_cost_per_hour": Decimal(
                str(metrics["estimated_cost_per_hour"])
            ),

            "estimated_latency_ms": Decimal(
                str(metrics["estimated_latency_ms"])
            ),

            "available_cpu_millicores": int(
                metrics["available_cpu_millicores"]
            ),

            "available_memory_mb": int(
                metrics["available_memory_mb"]
            ),

            # Obtain workload requirements from the actual request.
            "required_cpu_millicores": workload.cpu_millicores,
            "required_memory_mb": workload.memory_mb,

            # Preserve forecast provenance for experiment analysis.
            "prediction_metadata": {
                "zone": zone,
                "prediction_time": prediction["prediction_time"],
                "target_time": prediction["target_time"],
                "prediction_method": prediction_method,
                "model_type": prediction["model_type"],
                "input_data_source": prediction["input_data_source"],
                "forecast_horizon_hours": prediction[
                    "forecast_horizon_hours"
                ],
                "input_readings_estimated": prediction[
                    "input_readings_estimated"
                ],
            },
        })

    return candidates


# ============================================================
# SCHEDULING ORCHESTRATION
# ============================================================

@transaction.atomic
def schedule_workload(workload_id, candidates, weights=None):
    """
    Evaluate provider candidates, select the best eligible provider,
    and persist the scheduling decision and candidate evaluations.

    This function does not deploy workloads to cloud clusters.
    """

    workload = WorkloadRequest.objects.select_for_update().get(
        id=workload_id
    )

    if workload.status != "PENDING":
        raise ValueError(
            f"Workload cannot be scheduled from status: {workload.status}"
        )

    if not candidates:
        raise ValueError("At least one provider candidate is required.")

    weights = validate_weights(
        weights if weights is not None else DEFAULT_WEIGHTS
    )

    providers = [candidate.get("provider") for candidate in candidates]

    if len(providers) != len(set(providers)):
        raise ValueError(
            "Each provider can appear only once per decision."
        )

    for candidate in candidates:
        validate_candidate(candidate)

    evaluated_candidates = []
    eligible_candidates = []

    for candidate in candidates:
        eligible, reasons = evaluate_eligibility(workload, candidate)

        evaluated = {
            **candidate,
            "is_eligible": eligible,
            "eligibility_reasons": reasons,
        }

        evaluated_candidates.append(evaluated)

        if eligible:
            eligible_candidates.append(evaluated)

    ranked_candidates = calculate_candidate_scores(
        eligible_candidates,
        weights,
    )

    selected = ranked_candidates[0] if ranked_candidates else None

    if selected:
        decision_status = "COMPLETED"
        selected_provider = selected["provider"]
        selected_zone = PROVIDER_ZONES[selected_provider]

        decision_reason = (
            f"{selected_provider} selected with objective score "
            f"{selected['objective_score']:.6f}."
        )
    else:
        decision_status = "NO_ELIGIBLE_CANDIDATE"
        selected_provider = None
        selected_zone = None

        decision_reason = (
            "No provider satisfied all workload eligibility constraints."
        )

    decision = SchedulingDecision.objects.create(
        workload=workload,
        selected_provider=selected_provider,
        selected_zone=selected_zone,
        status=decision_status,
        decision_reason=decision_reason,
        scoring_weights={
            key: str(value)
            for key, value in weights.items()
        },
    )

    ranked_by_provider = {
        candidate["provider"]: candidate
        for candidate in ranked_candidates
    }

    for candidate in evaluated_candidates:
        ranked = ranked_by_provider.get(candidate["provider"])

        evaluation_details = {
            "eligibility_reasons": candidate["eligibility_reasons"],
            "scoring_method": "min_max_normalisation_weighted_sum",
            "scoring_weights": {
                key: str(value)
                for key, value in weights.items()
            },
            "score_available": ranked is not None,
        }

        # Preserve prediction provenance when candidates came from ML.
        if "prediction_metadata" in candidate:
            evaluation_details["prediction_metadata"] = (
                candidate["prediction_metadata"]
            )

        CandidateEvaluation.objects.create(
            decision=decision,
            provider=candidate["provider"],
            zone=PROVIDER_ZONES[candidate["provider"]],
            carbon_intensity=candidate["carbon_intensity"],
            estimated_cost_per_hour=candidate["estimated_cost_per_hour"],
            estimated_latency_ms=candidate["estimated_latency_ms"],
            available_cpu_millicores=candidate["available_cpu_millicores"],
            available_memory_mb=candidate["available_memory_mb"],
            objective_score=(
                ranked["objective_score"] if ranked else None
            ),
            is_eligible=candidate["is_eligible"],
            is_selected=(
                selected is not None
                and candidate["provider"] == selected_provider
            ),
            evaluation_details=evaluation_details,
        )

    return {
        "workload_id": workload.id,
        "decision_id": decision.id,
        "status": decision_status,
        "selected_provider": selected_provider,
        "selected_zone": selected_zone,
        "decision_reason": decision_reason,
        "ranked_candidates": ranked_candidates,
        "evaluated_candidates": evaluated_candidates,
    }
