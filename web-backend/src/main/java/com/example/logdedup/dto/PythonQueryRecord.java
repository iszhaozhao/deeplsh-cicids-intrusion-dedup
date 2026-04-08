package com.example.logdedup.dto;

import java.math.BigDecimal;

public record PythonQueryRecord(
    String querySampleId,
    String candidateSampleId,
    String queryLabel,
    String candidateLabel,
    Integer hashBucketHits,
    BigDecimal embeddingSimilarity,
    Integer isSameLabel,
    String sourceFile
) {
}
