package com.example.logdedup.dto;

public record ExperimentArtifactsResponse(
    String resultsDir,
    String baselineMetricsCsv,
    String bigruMetricsCsv,
    String summaryJson,
    String processedDataDir,
    String mlpModelArtifact,
    String bigruModelArtifact
) {
}
