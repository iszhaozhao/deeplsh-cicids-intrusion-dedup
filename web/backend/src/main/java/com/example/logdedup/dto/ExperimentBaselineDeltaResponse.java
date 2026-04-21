package com.example.logdedup.dto;

import java.math.BigDecimal;

public record ExperimentBaselineDeltaResponse(
    BigDecimal deltaF1,
    BigDecimal deltaRecall,
    BigDecimal deltaCompressionRate,
    BigDecimal deltaLatencyMs
) {
}
