package com.example.logdedup.service;

import com.example.logdedup.config.AppProperties;
import com.example.logdedup.dto.ExperimentMetricResponse;
import com.example.logdedup.dto.ExperimentSummaryResponse;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.BufferedReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Service;

@Service
public class ExperimentService {

    private static final String BASELINE_FILE = "cicids_baseline_metrics.csv";
    private static final String BIGRU_FILE = "cicids_bigru_metrics.csv";
    private static final String SUMMARY_FILE = "cicids_comparison_summary.json";
    private static final Set<String> CORE_MODELS = Set.of("baseline-mlp", "bigru-deeplsh");

    private final AppProperties appProperties;
    private final ObjectMapper objectMapper;

    public ExperimentService(AppProperties appProperties, ObjectMapper objectMapper) {
        this.appProperties = appProperties;
        this.objectMapper = objectMapper;
    }

    public List<ExperimentMetricResponse> listMetrics() {
        Path resultsDir = resultsDir();
        List<ExperimentMetricResponse> metrics = new ArrayList<>();
        metrics.addAll(readMetrics(resultsDir.resolve(BASELINE_FILE)));
        metrics.addAll(readMetrics(resultsDir.resolve(BIGRU_FILE)));
        return metrics.stream()
            .sorted(Comparator.comparing(ExperimentMetricResponse::f1, Comparator.nullsLast(Comparator.reverseOrder())))
            .toList();
    }

    public ExperimentSummaryResponse summary() {
        List<ExperimentMetricResponse> metrics = listMetrics();
        Path summaryPath = resultsDir().resolve(SUMMARY_FILE);
        if (!Files.exists(summaryPath)) {
            return emptySummary(metrics);
        }
        try {
            JsonNode root = objectMapper.readTree(summaryPath.toFile());
            ExperimentMetricResponse bestModel = metrics.stream()
                .max(Comparator.comparing(ExperimentMetricResponse::f1, Comparator.nullsLast(Comparator.naturalOrder())))
                .orElse(null);
            return new ExperimentSummaryResponse(
                !metrics.isEmpty(),
                root.path("results_dir").asText(resultsDir().toString()),
                root.path("top_k").isMissingNode() ? null : root.path("top_k").asInt(),
                root.path("sample_limit").isMissingNode() ? null : root.path("sample_limit").asInt(),
                bestModel == null ? null : bestModel.model(),
                bestModel == null ? null : bestModel.displayName(),
                bestModel == null ? null : bestModel.f1(),
                bestModel == null ? null : bestModel.precision(),
                bestModel == null ? null : bestModel.recall(),
                bestModel == null ? null : bestModel.compressionRate(),
                bestModel == null ? null : bestModel.avgQueryLatencyMs()
            );
        } catch (IOException ex) {
            return emptySummary(metrics);
        }
    }

    private List<ExperimentMetricResponse> readMetrics(Path csvPath) {
        if (!Files.exists(csvPath)) {
            return List.of();
        }
        try (BufferedReader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(reader)) {
            List<ExperimentMetricResponse> rows = new ArrayList<>();
            for (CSVRecord record : parser) {
                String model = record.get("model");
                if (!CORE_MODELS.contains(model)) {
                    continue;
                }
                rows.add(new ExperimentMetricResponse(
                    model,
                    displayName(model),
                    decimal(record.get("accuracy")),
                    decimal(record.get("precision")),
                    decimal(record.get("recall")),
                    decimal(record.get("f1")),
                    decimal(record.get("compression_rate")),
                    decimal(record.get("avg_query_latency_ms")),
                    decimal(record.get("threshold"))
                ));
            }
            return rows;
        } catch (IOException ex) {
            return List.of();
        }
    }

    private ExperimentSummaryResponse emptySummary(List<ExperimentMetricResponse> metrics) {
        ExperimentMetricResponse bestModel = metrics.stream()
            .max(Comparator.comparing(ExperimentMetricResponse::f1, Comparator.nullsLast(Comparator.naturalOrder())))
            .orElse(null);
        return new ExperimentSummaryResponse(
            !metrics.isEmpty(),
            resultsDir().toString(),
            null,
            null,
            bestModel == null ? null : bestModel.model(),
            bestModel == null ? null : bestModel.displayName(),
            bestModel == null ? null : bestModel.f1(),
            bestModel == null ? null : bestModel.precision(),
            bestModel == null ? null : bestModel.recall(),
            bestModel == null ? null : bestModel.compressionRate(),
            bestModel == null ? null : bestModel.avgQueryLatencyMs()
        );
    }

    private Path resultsDir() {
        return Path.of(appProperties.getExperiments().getResultsDir());
    }

    private BigDecimal decimal(String raw) {
        if (raw == null || raw.isBlank() || "null".equalsIgnoreCase(raw)) {
            return null;
        }
        return new BigDecimal(raw);
    }

    public String displayName(String model) {
        if (model == null || model.isBlank()) {
            return null;
        }
        return switch (model) {
            case "baseline-mlp", "mlp" -> "MLP + DeepLSH（baseline）";
            case "bigru-deeplsh", "bigru" -> "Bi-GRU + DeepLSH（论文主模型）";
            case "exact-md5" -> "Exact MD5";
            case "simhash" -> "SimHash";
            default -> model;
        };
    }
}
