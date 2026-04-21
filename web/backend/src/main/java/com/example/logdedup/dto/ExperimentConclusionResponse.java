package com.example.logdedup.dto;

public record ExperimentConclusionResponse(
    String headline,
    String summary,
    ExperimentBaselineDeltaResponse baselineDelta,
    String recommendation
) {
}
