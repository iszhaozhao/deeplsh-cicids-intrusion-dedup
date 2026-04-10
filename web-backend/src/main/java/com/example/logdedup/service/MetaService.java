package com.example.logdedup.service;

import com.example.logdedup.config.AppProperties;
import com.example.logdedup.dto.LabelStatResponse;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Service;

@Service
public class MetaService {

    private final AppProperties appProperties;

    public MetaService(AppProperties appProperties) {
        this.appProperties = appProperties;
    }

    public List<LabelStatResponse> listLabels() {
        try {
            List<LabelStatResponse> labels = runPythonLabels();
            if (!labels.isEmpty()) {
                return labels;
            }
        } catch (Exception ignored) {
            // fall through to CSV-based fallback
        }
        return readLabelsFromFlows();
    }

    private List<LabelStatResponse> runPythonLabels() throws IOException, InterruptedException {
        List<String> command = List.of("python3", appProperties.getPython().getScript(), "cicids-list-labels");
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.directory(Path.of(appProperties.getPython().getWorkdir()).toFile());
        processBuilder.redirectErrorStream(true);

        Process process = processBuilder.start();
        List<LabelStatResponse> labels = new ArrayList<>();
        try (BufferedReader reader = process.inputReader(StandardCharsets.UTF_8)) {
            reader.lines().forEach(line -> {
                if (!line.contains("\t")) {
                    return;
                }
                String[] parts = line.split("\t", 2);
                try {
                    labels.add(new LabelStatResponse(parts[0], Long.parseLong(parts[1])));
                } catch (NumberFormatException ignored) {
                    // skip malformed lines
                }
            });
        }
        process.waitFor();
        return labels;
    }

    private List<LabelStatResponse> readLabelsFromFlows() {
        Path flowsPath = Path.of(appProperties.getPython().getFlowsCsv());
        if (!Files.exists(flowsPath)) {
            return List.of();
        }
        try (BufferedReader reader = Files.newBufferedReader(flowsPath, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(reader)) {
            Map<String, Long> counts = parser.stream()
                .map(CSVRecord::toMap)
                .filter(row -> row.containsKey("Label"))
                .collect(Collectors.groupingBy(row -> row.get("Label"), Collectors.counting()));
            return counts.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue(Comparator.reverseOrder()))
                .map(entry -> new LabelStatResponse(entry.getKey(), entry.getValue()))
                .toList();
        } catch (IOException ex) {
            return List.of();
        }
    }
}
