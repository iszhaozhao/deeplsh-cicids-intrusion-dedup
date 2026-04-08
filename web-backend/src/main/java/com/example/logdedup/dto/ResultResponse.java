package com.example.logdedup.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.math.BigDecimal;

public record ResultResponse(
    Long id,
    Long taskId,
    Long logId,
    String attackType,
    @JsonProperty("hashCode") String hashCodeValue,
    BigDecimal similarityScore,
    String clusterId,
    Integer isRedundant,
    Long reserveLogId,
    String candidateSampleId,
    String sourceFile
) {
}
