package com.example.logdedup.dto;

import java.math.BigDecimal;

public record ExperimentMetricResponse(
    String model,
    String displayName,
    BigDecimal accuracy,
    BigDecimal precision,
    BigDecimal recall,
    BigDecimal f1,
    BigDecimal compressionRate,
    BigDecimal avgQueryLatencyMs,
    BigDecimal threshold
) {
}
