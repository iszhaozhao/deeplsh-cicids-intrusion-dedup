package com.example.logdedup.service;

import com.example.logdedup.dto.StatsOverviewResponse;
import com.example.logdedup.entity.DedupResult;
import com.example.logdedup.entity.DedupTask;
import com.example.logdedup.repository.DedupResultRepository;
import com.example.logdedup.repository.DedupTaskRepository;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.springframework.stereotype.Service;

@Service
public class StatsService {

    private final DedupTaskRepository taskRepository;
    private final DedupResultRepository resultRepository;
    private final ExperimentService experimentService;

    public StatsService(
        DedupTaskRepository taskRepository,
        DedupResultRepository resultRepository,
        ExperimentService experimentService
    ) {
        this.taskRepository = taskRepository;
        this.resultRepository = resultRepository;
        this.experimentService = experimentService;
    }

    public StatsOverviewResponse overview() {
        List<DedupTask> tasks = taskRepository.findAll();
        List<DedupResult> results = resultRepository.findAll();

        long totalLogs = tasks.stream().mapToLong(task -> task.getTotalLogs() == null ? 0 : task.getTotalLogs()).sum();
        BigDecimal avgCompressionRate = average(tasks.stream()
            .map(DedupTask::getCompressionRate)
            .filter(rate -> rate != null)
            .toList());
        BigDecimal avgLatency = average(tasks.stream()
            .map(DedupTask::getAvgLatencyMs)
            .filter(rate -> rate != null)
            .toList());

        List<Map<String, Object>> recentTasks = tasks.stream()
            .sorted((a, b) -> b.getCreateTime().compareTo(a.getCreateTime()))
            .limit(5)
            .map(task -> {
                Map<String, Object> map = new LinkedHashMap<>();
                map.put("id", task.getId());
                map.put("taskName", task.getTaskName());
                map.put("modelType", normalizeModelType(task.getModelType()));
                map.put("queryMode", task.getSampleId() != null && !task.getSampleId().isBlank() ? "sample_id" : "row_index");
                map.put("labelScope", task.getLabelScope());
                map.put("topK", task.getTopK());
                map.put("status", task.getStatus());
                map.put("compressionRate", task.getCompressionRate());
                map.put("avgLatencyMs", task.getAvgLatencyMs());
                return map;
            })
            .toList();

        Map<String, Long> attackCounts = results.stream()
            .collect(Collectors.groupingBy(DedupResult::getAttackType, LinkedHashMap::new, Collectors.counting()));

        List<Map<String, Object>> attackTypes = new ArrayList<>();
        attackCounts.forEach((type, count) -> {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("name", type);
            item.put("value", count);
            attackTypes.add(item);
        });

        DedupTask latestTask = tasks.stream()
            .sorted((a, b) -> b.getCreateTime().compareTo(a.getCreateTime()))
            .findFirst()
            .orElse(null);

        var summary = experimentService.summary();

        return new StatsOverviewResponse(
            tasks.size(),
            totalLogs,
            results.size(),
            tasks.stream().filter(task -> task.getStatus() != null && !"FAILED".equalsIgnoreCase(task.getStatus())).count(),
            avgCompressionRate,
            avgLatency,
            latestTask == null ? null : normalizeModelType(latestTask.getModelType()),
            latestTask == null ? null : latestTask.getLabelScope(),
            latestTask == null ? null : latestTask.getTopK(),
            summary.bestModelName(),
            summary.bestModelDisplayName(),
            summary.bestModelF1(),
            recentTasks,
            attackTypes
        );
    }

    private BigDecimal average(List<BigDecimal> values) {
        if (values.isEmpty()) {
            return BigDecimal.ZERO;
        }
        BigDecimal total = values.stream().reduce(BigDecimal.ZERO, BigDecimal::add);
        return total.divide(BigDecimal.valueOf(values.size()), 2, RoundingMode.HALF_UP);
    }

    private String normalizeModelType(String modelType) {
        if (modelType == null || modelType.isBlank()) {
            return "bigru";
        }
        return modelType.trim().toLowerCase();
    }
}
