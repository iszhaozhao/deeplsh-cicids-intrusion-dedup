package com.example.logdedup.dto;

import java.time.LocalDateTime;

public record TrainingJobResponse(
    Long id,
    String jobName,
    String jobType,
    String status,
    String runMessage,
    Integer exitCode,
    String logPath,
    LocalDateTime createTime,
    LocalDateTime startTime,
    LocalDateTime endTime,
    String paramsJson
) {
}

