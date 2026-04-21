package com.example.logdedup.dto;

import java.util.List;

public record ExperimentShowcaseResponse(
    boolean hasData,
    String datasetName,
    Long flowCount,
    Long pairCount,
    Integer topK,
    Integer sampleLimit,
    ExperimentModelShowcaseResponse bestModel,
    List<ExperimentModelShowcaseResponse> models,
    ExperimentConclusionResponse conclusion,
    ExperimentArtifactsResponse artifacts
) {
}
