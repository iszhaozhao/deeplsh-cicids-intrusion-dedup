package com.example.logdedup.dto;

import java.math.BigDecimal;

public record ExperimentSummaryResponse(
    boolean hasData,
    String resultsDir,
    Integer topK,
    Integer sampleLimit,
    String bestModelName,
    String bestModelDisplayName,
    BigDecimal bestModelF1,
    BigDecimal bestModelPrecision,
    BigDecimal bestModelRecall,
    BigDecimal bestModelCompressionRate,
    BigDecimal bestModelAvgQueryLatencyMs
) {
}
