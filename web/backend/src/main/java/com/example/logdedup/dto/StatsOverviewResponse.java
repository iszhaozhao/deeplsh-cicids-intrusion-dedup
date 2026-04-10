package com.example.logdedup.dto;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

public record StatsOverviewResponse(
    long totalTasks,
    long totalLogs,
    long totalResults,
    long recentQueryTasks,
    BigDecimal avgCompressionRate,
    BigDecimal avgLatencyMs,
    String latestModelType,
    String latestLabelScope,
    Integer latestTopK,
    String bestModelName,
    String bestModelDisplayName,
    BigDecimal bestModelF1,
    List<Map<String, Object>> recentTasks,
    List<Map<String, Object>> attackTypes
) {
}
