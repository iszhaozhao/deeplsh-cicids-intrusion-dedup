package com.example.logdedup.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.math.BigDecimal;

public record ResultResponse(
    Long id,
    Long taskId,
    Long logId,
    String attackType,
    String querySampleId,
    String queryLabel,
    String candidateLabel,
    Integer hashBucketHits,
    @JsonProperty("hashCode") String hashCodeValue,
    BigDecimal similarityScore,
    String clusterId,
    Integer isSameLabel,
    Integer isRedundant,
    Long reserveLogId,
    String candidateSampleId,
    String sourceFile
) {
}
