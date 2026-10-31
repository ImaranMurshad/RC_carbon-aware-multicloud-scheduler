from django.db import models


class CarbonReading(models.Model):

    ZONE_CHOICES = [
        ("IE", "Ireland"),
        ("DE", "Germany"),
        ("FR", "France"),
    ]

    zone = models.CharField(
        max_length=2,
        choices=ZONE_CHOICES
    )

    carbon_intensity = models.DecimalField(
        max_digits=10,
        decimal_places=3
    )

    timestamp = models.DateTimeField()

    updated_at = models.DateTimeField(
        null=True,
        blank=True
    )

    is_estimated = models.BooleanField(default=False)

    temporal_granularity = models.CharField(
        max_length=20,
        default="15_minutes"
    )

    raw_response = models.JSONField(
        null=True,
        blank=True
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]
        constraints = [
            models.UniqueConstraint(
                fields=["zone", "timestamp"],
                name="unique_carbon_reading_per_zone"
            )
        ]

    def __str__(self):
        return f"{self.zone} - {self.carbon_intensity} gCO2e/kWh"