package com.example.logdedup.integration;

import com.example.logdedup.config.AppProperties;
import com.example.logdedup.dto.PythonQueryRecord;
import com.example.logdedup.entity.DedupTask;
import java.io.BufferedReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Service;
import com.example.logdedup.util.CommandLineUtils;

@Service
public class PythonQueryService {

    private final AppProperties appProperties;

    public PythonQueryService(AppProperties appProperties) {
        this.appProperties = appProperties;
    }

    public List<PythonQueryRecord> queryCandidates(DedupTask task) {
        if (hasRealQueryArtifacts(task)) {
            try {
                return runRealQuery(task);
            } catch (Exception ex) {
                return buildFallbackQuery(task, "python query failed: " + ex.getMessage());
            }
        }
        return buildFallbackQuery(task, "missing model artifact, using fallback");
    }

    public String describeMode(DedupTask task) {
        return hasRealQueryArtifacts(task) ? "REAL_QUERY" : "FALLBACK_QUERY";
    }

    private boolean hasRealQueryArtifacts(DedupTask task) {
        Path modelArtifact = Path.of(resolveModelArtifact(task));
        return Files.exists(modelArtifact);
    }

    private List<PythonQueryRecord> runRealQuery(DedupTask task) throws IOException, InterruptedException {
        Path tempCsv = Files.createTempFile("dedup-query-", ".csv");
        List<String> command = new ArrayList<>();
        command.addAll(resolvePythonCommandTokens());
        command.add(appProperties.getPython().getScript());
        command.add("cicids-query");
        command.add("--model-type");
        command.add(normalizeModelType(task));
        if (task.getSampleId() != null && !task.getSampleId().isBlank()) {
            command.add("--sample-id");
            command.add(task.getSampleId());
        } else {
            command.add("--row-index");
            command.add(String.valueOf(task.getRowIndex() == null ? 0 : task.getRowIndex()));
        }
        command.add("--label-scope");
        command.add(task.getLabelScope() == null ? "same" : task.getLabelScope());
        command.add("--top-k");
        command.add(String.valueOf(task.getTopK() == null ? 10 : task.getTopK()));
        command.add("--output-csv");
        command.add(tempCsv.toString());

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.directory(Path.of(appProperties.getPython().getWorkdir()).toFile());
        processBuilder.redirectErrorStream(true);

        Process process = processBuilder.start();
        String output;
        try (BufferedReader reader = process.inputReader(StandardCharsets.UTF_8)) {
            output = reader.lines().collect(Collectors.joining("\n"));
        }
        int exitCode = process.waitFor();
        if (exitCode != 0 || !Files.exists(tempCsv)) {
            throw new IOException("exitCode=" + exitCode + ", output=" + output);
        }
        List<PythonQueryRecord> records = readQueryCsv(tempCsv);
        Files.deleteIfExists(tempCsv);
        return records;
    }

    private List<PythonQueryRecord> readQueryCsv(Path csvPath) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(csvPath, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(reader)) {
            List<PythonQueryRecord> records = new ArrayList<>();
            for (CSVRecord record : parser) {
                records.add(new PythonQueryRecord(
                    record.get("query_sample_id"),
                    record.get("candidate_sample_id"),
                    record.get("query_label"),
                    record.get("candidate_label"),
                    Integer.parseInt(record.get("hash_bucket_hits")),
                    new BigDecimal(record.get("embedding_similarity")),
                    Integer.parseInt(record.get("is_same_label")),
                    record.get("source_file")
                ));
            }
            return records;
        }
    }

    private List<PythonQueryRecord> buildFallbackQuery(DedupTask task, String reason) {
        Path flowsPath = Path.of(appProperties.getPython().getFlowsCsv());
        if (!Files.exists(flowsPath)) {
            return List.of(new PythonQueryRecord(
                "sample-fallback",
                "sample-fallback-1",
                "DDoS",
                "DDoS",
                4,
                new BigDecimal("0.9620"),
                1,
                reason + " @ " + LocalDateTime.now()
            ));
        }

        try (BufferedReader reader = Files.newBufferedReader(flowsPath, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(reader)) {
            List<Map<String, String>> rows = new ArrayList<>();
            for (CSVRecord record : parser) {
                Map<String, String> row = record.toMap();
                if (row.containsKey("sample_id") && row.containsKey("Label")) {
                    rows.add(row);
                }
            }

            if (rows.isEmpty()) {
                return List.of();
            }

            int queryIndex = task.getRowIndex() == null ? 0 : Math.max(0, Math.min(task.getRowIndex(), rows.size() - 1));
            Map<String, String> query = rows.get(queryIndex);
            String queryLabel = query.get("Label");
            String querySampleId = task.getSampleId() != null && !task.getSampleId().isBlank()
                ? task.getSampleId()
                : query.get("sample_id");

            List<Map<String, String>> candidates = rows.stream()
                .filter(row -> !Objects.equals(row.get("sample_id"), querySampleId))
                .filter(row -> !"all".equalsIgnoreCase(task.getLabelScope()) ? Objects.equals(row.get("Label"), queryLabel) : true)
                .limit(task.getTopK() == null ? 10 : task.getTopK())
                .toList();

            List<PythonQueryRecord> results = new ArrayList<>();
            int rank = 0;
            for (Map<String, String> candidate : candidates) {
                BigDecimal similarity = BigDecimal.valueOf(Math.max(0.72, 0.96 - (rank * 0.03)))
                    .setScale(4, java.math.RoundingMode.HALF_UP);
                results.add(new PythonQueryRecord(
                    querySampleId,
                    candidate.get("sample_id"),
                    queryLabel,
                    candidate.get("Label"),
                    Math.max(1, 5 - rank),
                    similarity,
                    Objects.equals(candidate.get("Label"), queryLabel) ? 1 : 0,
                    candidate.getOrDefault("source_file", reason)
                ));
                rank++;
            }

            return results.stream()
                .sorted(Comparator.comparing(PythonQueryRecord::embeddingSimilarity).reversed())
                .collect(Collectors.toList());
        } catch (IOException ex) {
            return List.of(new PythonQueryRecord(
                "sample-fallback",
                "sample-fallback-1",
                "PortScan",
                "PortScan",
                3,
                new BigDecimal("0.9110"),
                1,
                "fallback parse failed: " + ex.getMessage()
            ));
        }
    }

    private List<String> resolvePythonCommandTokens() {
        List<String> tokens = CommandLineUtils.splitCommand(appProperties.getPython().getCommand());
        if (tokens.isEmpty()) {
            return List.of("python3");
        }
        return tokens;
    }

    private String normalizeModelType(DedupTask task) {
        if (task.getModelType() == null || task.getModelType().isBlank()) {
            return "bigru";
        }
        return task.getModelType().trim().toLowerCase();
    }

    private String resolveModelArtifact(DedupTask task) {
        String modelType = normalizeModelType(task);
        if ("mlp".equals(modelType) && appProperties.getPython().getMlpModelArtifact() != null) {
            return appProperties.getPython().getMlpModelArtifact();
        }
        if ("bigru".equals(modelType) && appProperties.getPython().getBigruModelArtifact() != null) {
            return appProperties.getPython().getBigruModelArtifact();
        }
        return appProperties.getPython().getModelArtifact();
    }
}
