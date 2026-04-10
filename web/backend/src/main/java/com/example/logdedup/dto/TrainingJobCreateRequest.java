package com.example.logdedup.dto;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.validation.constraints.NotBlank;

public record TrainingJobCreateRequest(
    String jobName,
    @NotBlank String jobType,
    JsonNode params
) {
}

