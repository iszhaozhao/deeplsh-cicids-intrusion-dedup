package com.example.logdedup.service;

import com.example.logdedup.config.AppProperties;
import com.example.logdedup.dto.ExperimentArtifactsResponse;
import com.example.logdedup.dto.ExperimentBaselineDeltaResponse;
import com.example.logdedup.dto.ExperimentConclusionResponse;
import com.example.logdedup.dto.ExperimentMetricResponse;
import com.example.logdedup.dto.ExperimentModelShowcaseResponse;
import com.example.logdedup.dto.ExperimentShowcaseResponse;
import com.example.logdedup.dto.ExperimentSummaryResponse;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.BufferedReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
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
    private static final Comparator<ExperimentMetricResponse> METRIC_RANKING = Comparator
        .comparing(ExperimentMetricResponse::f1, Comparator.nullsLast(Comparator.naturalOrder()))
        .thenComparing(ExperimentMetricResponse::recall, Comparator.nullsLast(Comparator.naturalOrder()))
        .thenComparing(ExperimentMetricResponse::precision, Comparator.nullsLast(Comparator.naturalOrder()));

    private final AppProperties appProperties;
    private final ObjectMapper objectMapper;

    public ExperimentService(AppProperties appProperties, ObjectMapper objectMapper) {
        this.appProperties = appProperties;
        this.objectMapper = objectMapper;
    }

    public List<ExperimentMetricResponse> listMetrics() {
        List<ExperimentMetricResponse> metrics = listAllMetrics();
        return metrics.stream()
            .filter(item -> CORE_MODELS.contains(item.model()))
            .sorted(METRIC_RANKING.reversed())
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
                .max(METRIC_RANKING)
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

    public ExperimentShowcaseResponse showcase() {
        List<ExperimentMetricResponse> allMetrics = listAllMetrics();
        ExperimentMetricResponse bestModel = allMetrics.stream().max(METRIC_RANKING).orElse(null);
        ExperimentMetricResponse baselineModel = allMetrics.stream()
            .filter(item -> "baseline-mlp".equals(item.model()))
            .findFirst()
            .orElse(null);

        Integer topK = null;
        Integer sampleLimit = null;
        String baselineMetricsCsv = resultsDir().resolve(BASELINE_FILE).toString();
        String bigruMetricsCsv = resultsDir().resolve(BIGRU_FILE).toString();
        String summaryJson = resultsDir().resolve(SUMMARY_FILE).toString();
        if (Files.exists(resultsDir().resolve(SUMMARY_FILE))) {
            try {
                JsonNode root = objectMapper.readTree(resultsDir().resolve(SUMMARY_FILE).toFile());
                topK = root.path("top_k").isMissingNode() ? null : root.path("top_k").asInt();
                sampleLimit = root.path("sample_limit").isMissingNode() ? null : root.path("sample_limit").asInt();
                baselineMetricsCsv = root.path("baseline_metrics_csv").asText(baselineMetricsCsv);
                bigruMetricsCsv = root.path("bigru_metrics_csv").asText(bigruMetricsCsv);
                summaryJson = resultsDir().resolve(SUMMARY_FILE).toString();
            } catch (IOException ignored) {
                // Fall back to defaults based on file layout.
            }
        }

        Path processedDir = processedDataDir();
        Long flowCount = readMetadataCount(processedDir.resolve("metadata.json"), "n_flows");
        Long pairCount = readMetadataCount(processedDir.resolve("metadata.json"), "n_pairs");

        List<ExperimentModelShowcaseResponse> models = allMetrics.stream()
            .sorted(METRIC_RANKING.reversed())
            .map(item -> toShowcaseModel(item, bestModel))
            .toList();

        return new ExperimentShowcaseResponse(
            !models.isEmpty(),
            "CIC-IDS-2017",
            flowCount,
            pairCount,
            topK,
            sampleLimit,
            toShowcaseModel(bestModel, bestModel),
            models,
            buildConclusion(bestModel, baselineModel),
            new ExperimentArtifactsResponse(
                resultsDir().toString(),
                baselineMetricsCsv,
                bigruMetricsCsv,
                summaryJson,
                processedDir.toString(),
                appProperties.getPython().getMlpModelArtifact(),
                appProperties.getPython().getBigruModelArtifact()
            )
        );
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
            .max(METRIC_RANKING)
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

    private Path processedDataDir() {
        return repoRoot().resolve("datasets").resolve("cicids").resolve("processed").resolve("full");
    }

    private Path repoRoot() {
        Path resultsDir = resultsDir().toAbsolutePath().normalize();
        Path current = resultsDir;
        for (int i = 0; i < 4 && current != null; i++) {
            current = current.getParent();
        }
        return current == null ? resultsDir : current;
    }

    private BigDecimal decimal(String raw) {
        if (raw == null || raw.isBlank() || "null".equalsIgnoreCase(raw)) {
            return null;
        }
        return new BigDecimal(raw);
    }

    private List<ExperimentMetricResponse> listAllMetrics() {
        Path resultsDir = resultsDir();
        List<ExperimentMetricResponse> metrics = new ArrayList<>();
        metrics.addAll(readMetrics(resultsDir.resolve(BASELINE_FILE)));
        metrics.addAll(readMetrics(resultsDir.resolve(BIGRU_FILE)));
        return metrics;
    }

    private Long readMetadataCount(Path metadataPath, String field) {
        if (!Files.exists(metadataPath)) {
            return null;
        }
        try {
            JsonNode root = objectMapper.readTree(metadataPath.toFile());
            return root.path(field).isMissingNode() ? null : root.path(field).asLong();
        } catch (IOException ex) {
            return null;
        }
    }

    private ExperimentModelShowcaseResponse toShowcaseModel(
        ExperimentMetricResponse metric,
        ExperimentMetricResponse bestModel
    ) {
        if (metric == null) {
            return null;
        }
        return new ExperimentModelShowcaseResponse(
            metric.model(),
            metric.displayName(),
            metric.accuracy(),
            metric.precision(),
            metric.recall(),
            metric.f1(),
            metric.compressionRate(),
            metric.avgQueryLatencyMs(),
            metric.threshold(),
            bestModel != null && metric.model().equals(bestModel.model())
        );
    }

    private ExperimentConclusionResponse buildConclusion(
        ExperimentMetricResponse bestModel,
        ExperimentMetricResponse baselineModel
    ) {
        if (bestModel == null) {
            return new ExperimentConclusionResponse(
                "尚未检测到 full 口径实验结果",
                "请先确认 artifacts/cicids/results/full 目录中的指标文件已经准备完成。",
                null,
                "建议先执行 CIC-IDS-2017 full 口径的离线评估，再进入答辩展示页。"
            );
        }

        String headline = "%s 是当前 full 口径下的最佳方案".formatted(bestModel.displayName());
        String summary = "%s 以 F1 为主指标领先，同时兼顾 Recall 与 Precision，适合作为论文主结论与系统展示口径。"
            .formatted(bestModel.displayName());
        ExperimentBaselineDeltaResponse baselineDelta = baselineModel == null ? null : new ExperimentBaselineDeltaResponse(
            subtract(bestModel.f1(), baselineModel.f1()),
            subtract(bestModel.recall(), baselineModel.recall()),
            subtract(bestModel.compressionRate(), baselineModel.compressionRate()),
            subtract(bestModel.avgQueryLatencyMs(), baselineModel.avgQueryLatencyMs())
        );

        String recommendation = recommendation(bestModel, baselineDelta);
        return new ExperimentConclusionResponse(headline, summary, baselineDelta, recommendation);
    }

    private String recommendation(ExperimentMetricResponse bestModel, ExperimentBaselineDeltaResponse baselineDelta) {
        if ("bigru-deeplsh".equals(bestModel.model()) && baselineDelta != null) {
            String latencyText = baselineDelta.deltaLatencyMs() == null
                ? "时延处于可接受范围"
                : (baselineDelta.deltaLatencyMs().signum() < 0 ? "查询时延也优于 MLP" : "虽然查询时延略高于 MLP，但仍可满足本地演示与答辩展示");
            return "推荐在正式成果展示中采用 Bi-GRU + DeepLSH 作为主模型，因为它在 F1、Recall 和压缩率上整体最优，%s。".formatted(latencyText);
        }
        if ("baseline-mlp".equals(bestModel.model())) {
            return "推荐将 MLP + DeepLSH 作为轻量基线方案，用于强调深度哈希方法相对传统规则方法的稳定提升。";
        }
        if ("simhash".equals(bestModel.model())) {
            return "推荐将 SimHash 作为传统近似去重基线进行对比，突出深度表示学习在复杂攻击模式下的优势。";
        }
        return "推荐将当前最佳模型作为论文主模型，其余方法作为基线与消融对照。";
    }

    private BigDecimal subtract(BigDecimal left, BigDecimal right) {
        if (left == null || right == null) {
            return null;
        }
        return left.subtract(right).setScale(4, RoundingMode.HALF_UP);
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
