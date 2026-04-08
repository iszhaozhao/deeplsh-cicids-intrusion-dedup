package com.example.logdedup.dto;

import java.math.BigDecimal;
import java.time.LocalDateTime;

public record TaskResponse(
    Long id,
    String taskName,
    BigDecimal similarityThreshold,
    Integer timeWindow,
    String reservePolicy,
    Integer hashBits,
    Integer totalLogs,
    Integer redundantLogs,
    BigDecimal compressionRate,
    BigDecimal avgLatencyMs,
    String status,
    String sampleId,
    Integer rowIndex,
    String labelScope,
    Integer topK,
    String runMessage,
    LocalDateTime createTime
) {
}
