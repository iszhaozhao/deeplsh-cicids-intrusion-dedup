package com.example.logdedup.dto;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

public record StatsOverviewResponse(
    long totalTasks,
    long totalLogs,
    long totalResults,
    BigDecimal avgCompressionRate,
    BigDecimal avgLatencyMs,
    List<Map<String, Object>> recentTasks,
    List<Map<String, Object>> attackTypes
) {
}
