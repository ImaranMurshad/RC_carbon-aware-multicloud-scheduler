
from django.db import models


class WorkloadRequest(models.Model):

    PRIORITY_CHOICES = [
        ("LOW", "Low"),
        ("MEDIUM", "Medium"),
        ("HIGH", "High"),
        ("CRITICAL", "Critical"),
    ]

    STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("SCHEDULED", "Scheduled"),
        ("DEPLOYED", "Deployed"),
        ("FAILED", "Failed"),
    ]

    name = models.CharField(max_length=150)
    container_image = models.CharField(max_length=255)
    cpu_millicores = models.PositiveIntegerField()
    memory_mb = models.PositiveIntegerField()

    max_latency_ms = models.PositiveIntegerField(
        null=True,
        blank=True
    )

    max_cost_per_hour = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True
    )

    priority = models.CharField(
        max_length=10,
        choices=PRIORITY_CHOICES,
        default="MEDIUM"
    )

    status = models.CharField(
        max_length=15,
        choices=STATUS_CHOICES,
        default="PENDING"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.status})"


class SchedulingDecision(models.Model):

    PROVIDER_CHOICES = [
        ("AWS", "Amazon Web Services"),
        ("AZURE", "Microsoft Azure"),
        ("GCP", "Google Cloud Platform"),
    ]

    STATUS_CHOICES = [
        ("COMPLETED", "Completed"),
        ("NO_ELIGIBLE_CANDIDATE", "No Eligible Candidate"),
    ]

    workload = models.ForeignKey(
        WorkloadRequest,
        on_delete=models.CASCADE,
        related_name="scheduling_decisions"
    )

    selected_provider = models.CharField(
        max_length=10,
        choices=PROVIDER_CHOICES,
        null=True,
        blank=True
    )

    selected_zone = models.CharField(
        max_length=2,
        null=True,
        blank=True
    )

    status = models.CharField(
        max_length=30,
        choices=STATUS_CHOICES,
        default="COMPLETED"
    )

    decision_reason = models.TextField(blank=True)
    scoring_weights = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Decision for {self.workload.name}"


class CandidateEvaluation(models.Model):

    PROVIDER_CHOICES = [
        ("AWS", "Amazon Web Services"),
        ("AZURE", "Microsoft Azure"),
        ("GCP", "Google Cloud Platform"),
    ]

    decision = models.ForeignKey(
        SchedulingDecision,
        on_delete=models.CASCADE,
        related_name="candidates"
    )

    provider = models.CharField(
        max_length=10,
        choices=PROVIDER_CHOICES
    )

    zone = models.CharField(max_length=2)

    carbon_intensity = models.DecimalField(
        max_digits=10,
        decimal_places=3,
        null=True,
        blank=True
    )

    estimated_cost_per_hour = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True
    )

    estimated_latency_ms = models.DecimalField(
        max_digits=10,
        decimal_places=3,
        null=True,
        blank=True
    )

    available_cpu_millicores = models.PositiveIntegerField(
        null=True,
        blank=True
    )

    available_memory_mb = models.PositiveIntegerField(
        null=True,
        blank=True
    )

    objective_score = models.DecimalField(
        max_digits=12,
        decimal_places=6,
        null=True,
        blank=True
    )

    is_eligible = models.BooleanField(default=False)
    is_selected = models.BooleanField(default=False)
    evaluation_details = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["objective_score"]

        constraints = [
            models.UniqueConstraint(
                fields=["decision", "provider"],
                name="unique_provider_per_scheduling_decision"
            )
        ]

    def __str__(self):
        return f"{self.provider} candidate for decision {self.decision_id}"
