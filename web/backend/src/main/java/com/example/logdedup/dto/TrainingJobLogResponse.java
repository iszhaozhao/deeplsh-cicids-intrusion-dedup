package com.example.logdedup.dto;

import java.util.List;

public record TrainingJobLogResponse(
    Long jobId,
    String status,
    String runMessage,
    Integer exitCode,
    List<String> lines
) {
}

