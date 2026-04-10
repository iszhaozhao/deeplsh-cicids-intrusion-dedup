package com.example.logdedup.dto;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.math.BigDecimal;

public record TaskCreateRequest(
    @NotBlank String taskName,
    @NotBlank String modelType,
    @NotNull @DecimalMin("0.10") @DecimalMax("1.00") BigDecimal similarityThreshold,
    @NotNull Integer timeWindow,
    @NotBlank String reservePolicy,
    @NotNull Integer hashBits,
    String sampleId,
    Integer rowIndex,
    String labelScope,
    Integer topK
) {
}
