package com.example.logdedup.service;

import com.example.logdedup.config.AppProperties;
import com.example.logdedup.dto.TrainingJobCreateRequest;
import com.example.logdedup.dto.TrainingJobLogResponse;
import com.example.logdedup.dto.TrainingJobResponse;
import com.example.logdedup.entity.TrainingJob;
import com.example.logdedup.repository.TrainingJobRepository;
import com.example.logdedup.util.CommandLineUtils;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PreDestroy;
import java.io.RandomAccessFile;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

@Service
public class TrainingJobService {

    private static final int MAX_TAIL_LINES = 400;

    private final TrainingJobRepository trainingJobRepository;
    private final AppProperties appProperties;
    private final ObjectMapper objectMapper;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    public TrainingJobService(
        TrainingJobRepository trainingJobRepository,
        AppProperties appProperties,
        ObjectMapper objectMapper
    ) {
        this.trainingJobRepository = trainingJobRepository;
        this.appProperties = appProperties;
        this.objectMapper = objectMapper;
    }

    public List<TrainingJobResponse> listJobs() {
        return trainingJobRepository.findAll().stream()
            .sorted((a, b) -> b.getCreateTime().compareTo(a.getCreateTime()))
            .map(this::toResponse)
            .toList();
    }

    public TrainingJobResponse getJob(Long id) {
        return trainingJobRepository.findById(id)
            .map(this::toResponse)
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "训练任务不存在"));
    }

    public TrainingJobResponse createJob(TrainingJobCreateRequest request) {
        String jobType = normalizeJobType(request.jobType());
        validateJobType(jobType);

        TrainingJob job = new TrainingJob();
        job.setJobName(Optional.ofNullable(request.jobName()).filter(v -> !v.isBlank()).orElse(jobType));
        job.setJobType(jobType);
        job.setStatus("PENDING");
        job.setRunMessage("Job created");
        job.setCreateTime(LocalDateTime.now());
        job.setParamsJson(toJson(request.params()));
        TrainingJob saved = trainingJobRepository.save(job);
        return toResponse(saved);
    }

    public TrainingJobResponse startJob(Long id) {
        TrainingJob job = trainingJobRepository.findById(id)
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "训练任务不存在"));

        if (Objects.equals(job.getStatus(), "RUNNING")) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "任务正在运行中");
        }
        if (!trainingJobRepository.findByStatus("RUNNING").isEmpty()) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "已有训练任务在运行中，请等待结束后再启动");
        }

        Path logPath = allocateLogPath(job.getId());
        job.setLogPath(logPath.toString());
        job.setStatus("RUNNING");
        job.setRunMessage("Launching python pipeline");
        job.setStartTime(LocalDateTime.now());
        job.setEndTime(null);
        job.setExitCode(null);
        trainingJobRepository.save(job);

        List<String> command = buildPythonCommand(job);
        executor.submit(() -> runJob(job.getId(), command, logPath));
        return toResponse(job);
    }

    public TrainingJobLogResponse tailLogs(Long id, Integer tail) {
        TrainingJob job = trainingJobRepository.findById(id)
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "训练任务不存在"));

        int limit = tail == null ? 200 : Math.max(1, Math.min(MAX_TAIL_LINES, tail));
        List<String> lines = readTailLines(job.getLogPath(), limit);
        return new TrainingJobLogResponse(job.getId(), job.getStatus(), job.getRunMessage(), job.getExitCode(), lines);
    }

    private void runJob(Long jobId, List<String> command, Path logPath) {
        TrainingJob job = trainingJobRepository.findById(jobId).orElse(null);
        if (job == null) {
            return;
        }

        try {
            Files.createDirectories(logPath.getParent());
            try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(logPath, StandardCharsets.UTF_8))) {
                writer.println("started_at=" + LocalDateTime.now());
                writer.println("workdir=" + appProperties.getPython().getWorkdir());
                writer.println("command=" + String.join(" ", command));
                writer.println("----");
            }

            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.directory(Path.of(appProperties.getPython().getWorkdir()).toFile());
            processBuilder.redirectErrorStream(true);
            processBuilder.redirectOutput(ProcessBuilder.Redirect.appendTo(logPath.toFile()));

            Process process = processBuilder.start();
            int exitCode = process.waitFor();

            TrainingJob refreshed = trainingJobRepository.findById(jobId).orElse(null);
            if (refreshed == null) {
                return;
            }
            refreshed.setExitCode(exitCode);
            refreshed.setEndTime(LocalDateTime.now());
            refreshed.setStatus(exitCode == 0 ? "SUCCESS" : "FAILED");
            refreshed.setRunMessage("exitCode=" + exitCode);
            trainingJobRepository.save(refreshed);
        } catch (Exception ex) {
            TrainingJob refreshed = trainingJobRepository.findById(jobId).orElse(null);
            if (refreshed == null) {
                return;
            }
            refreshed.setExitCode(-1);
            refreshed.setEndTime(LocalDateTime.now());
            refreshed.setStatus("FAILED");
            refreshed.setRunMessage("failed: " + ex.getMessage());
            trainingJobRepository.save(refreshed);
            try {
                Files.writeString(logPath, "\nERROR: " + ex.getMessage() + "\n", StandardCharsets.UTF_8, java.nio.file.StandardOpenOption.APPEND);
            } catch (IOException ignored) {
            }
        }
    }

    private List<String> buildPythonCommand(TrainingJob job) {
        List<String> tokens = CommandLineUtils.splitCommand(appProperties.getPython().getCommand());
        if (tokens.isEmpty()) {
            tokens = List.of("python3");
        }
        List<String> command = new ArrayList<>(tokens);
        command.add(appProperties.getPython().getScript());

        String jobType = normalizeJobType(job.getJobType());
        JsonNode params = parseJson(job.getParamsJson());
        switch (jobType) {
            case "cicids-prepare" -> buildCicidsPrepareArgs(command, params);
            case "cicids-train-mlp" -> buildCicidsTrainMlpArgs(command, params);
            case "cicids-train-bigru" -> buildCicidsTrainBigruArgs(command, params);
            case "cicids-eval" -> buildCicidsEvalArgs(command, params);
            default -> throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "未知 jobType: " + jobType);
        }
        return command;
    }

    private void buildCicidsPrepareArgs(List<String> command, JsonNode params) {
        command.add("cicids-prepare");
        String dataRepo = readString(params, "dataRepo").orElse("./datasets/cicids/raw");
        String outputDir = readString(params, "outputDir").orElse("./datasets/cicids/processed/full");
        int maxSamples = readInt(params, "maxSamples").orElse(12000);
        int maxPairs = readInt(params, "maxPairs").orElse(20000);
        int seed = readInt(params, "seed").orElse(42);

        command.add("--data-repo");
        command.add(dataRepo);
        command.add("--output-dir");
        command.add(outputDir);
        command.add("--max-samples");
        command.add(String.valueOf(maxSamples));
        command.add("--max-pairs");
        command.add(String.valueOf(maxPairs));
        command.add("--seed");
        command.add(String.valueOf(seed));
    }

    private void buildCicidsTrainMlpArgs(List<String> command, JsonNode params) {
        command.add("cicids-train-mlp");
        String dataRepo = readString(params, "dataRepo").orElse("./datasets/cicids/raw");
        String outputDir = readString(params, "outputDir").orElse("./datasets/cicids/processed/full");
        int maxSamples = readInt(params, "maxSamples").orElse(12000);
        int maxPairs = readInt(params, "maxPairs").orElse(20000);
        int epochs = readInt(params, "epochs").orElse(10);
        int batchSize = readInt(params, "batchSize").orElse(256);
        int seed = readInt(params, "seed").orElse(42);

        command.add("--data-repo");
        command.add(dataRepo);
        command.add("--output-dir");
        command.add(outputDir);
        command.add("--max-samples");
        command.add(String.valueOf(maxSamples));
        command.add("--max-pairs");
        command.add(String.valueOf(maxPairs));
        command.add("--epochs");
        command.add(String.valueOf(epochs));
        command.add("--batch-size");
        command.add(String.valueOf(batchSize));
        command.add("--seed");
        command.add(String.valueOf(seed));
    }

    private void buildCicidsTrainBigruArgs(List<String> command, JsonNode params) {
        command.add("cicids-train-bigru");
        String dataRepo = readString(params, "dataRepo").orElse("./datasets/cicids/raw");
        String outputDir = readString(params, "outputDir").orElse("./datasets/cicids/processed/full");
        int maxSamples = readInt(params, "maxSamples").orElse(12000);
        int maxPairs = readInt(params, "maxPairs").orElse(20000);
        int epochs = readInt(params, "epochs").orElse(10);
        int batchSize = readInt(params, "batchSize").orElse(128);
        int seed = readInt(params, "seed").orElse(42);
        int embedDim = readInt(params, "embedDim").orElse(64);
        int gruUnits = readInt(params, "gruUnits").orElse(64);
        int denseDim = readInt(params, "denseDim").orElse(128);

        command.add("--data-repo");
        command.add(dataRepo);
        command.add("--output-dir");
        command.add(outputDir);
        command.add("--max-samples");
        command.add(String.valueOf(maxSamples));
        command.add("--max-pairs");
        command.add(String.valueOf(maxPairs));
        command.add("--epochs");
        command.add(String.valueOf(epochs));
        command.add("--batch-size");
        command.add(String.valueOf(batchSize));
        command.add("--seed");
        command.add(String.valueOf(seed));
        command.add("--embed-dim");
        command.add(String.valueOf(embedDim));
        command.add("--gru-units");
        command.add(String.valueOf(gruUnits));
        command.add("--dense-dim");
        command.add(String.valueOf(denseDim));
    }

    private void buildCicidsEvalArgs(List<String> command, JsonNode params) {
        command.add("cicids-eval");
        String outputDir = readString(params, "outputDir").orElse("./datasets/cicids/processed/full");
        String resultsDir = readString(params, "resultsDir").orElse(appProperties.getExperiments().getResultsDir());
        int topK = readInt(params, "topK").orElse(10);
        int sampleLimit = readInt(params, "sampleLimit").orElse(50);

        command.add("--output-dir");
        command.add(outputDir);
        command.add("--results-dir");
        command.add(resultsDir);
        command.add("--top-k");
        command.add(String.valueOf(topK));
        command.add("--sample-limit");
        command.add(String.valueOf(sampleLimit));
    }

    private Path allocateLogPath(Long jobId) {
        Path base = Path.of(appProperties.getPython().getWorkdir()).resolve("artifacts").resolve("logs").resolve("train-logs");
        return base.resolve("job-" + jobId + ".log");
    }

    private List<String> readTailLines(String logPath, int limit) {
        if (logPath == null || logPath.isBlank()) {
            return List.of();
        }
        Path path = Path.of(logPath);
        if (!Files.exists(path)) {
            return List.of();
        }
        try {
            return tail(path, limit);
        } catch (IOException ex) {
            return List.of("ERROR: unable to read log: " + ex.getMessage());
        }
    }

    private List<String> tail(Path path, int lines) throws IOException {
        long size = Files.size(path);
        int chunk = 64 * 1024;
        long start = Math.max(0, size - chunk);
        byte[] buf;
        try (RandomAccessFile raf = new RandomAccessFile(path.toFile(), "r")) {
            raf.seek(start);
            buf = new byte[(int) (size - start)];
            raf.readFully(buf);
        }
        String content = new String(buf, StandardCharsets.UTF_8);
        String[] rawLines = content.split("\\R");
        Deque<String> tail = new ArrayDeque<>();
        for (String line : rawLines) {
            tail.addLast(line);
            if (tail.size() > lines) {
                tail.removeFirst();
            }
        }
        return List.copyOf(tail);
    }

    private String toJson(JsonNode node) {
        if (node == null || node.isNull()) {
            return "{}";
        }
        try {
            return objectMapper.writeValueAsString(node);
        } catch (JsonProcessingException ex) {
            return "{}";
        }
    }

    private JsonNode parseJson(String json) {
        if (json == null || json.isBlank()) {
            return objectMapper.createObjectNode();
        }
        try {
            return objectMapper.readTree(json);
        } catch (JsonProcessingException ex) {
            return objectMapper.createObjectNode();
        }
    }

    private Optional<String> readString(JsonNode params, String key) {
        if (params == null || params.isNull()) {
            return Optional.empty();
        }
        JsonNode node = params.get(key);
        if (node == null || node.isNull() || node.asText().isBlank()) {
            return Optional.empty();
        }
        return Optional.of(node.asText());
    }

    private Optional<Integer> readInt(JsonNode params, String key) {
        if (params == null || params.isNull()) {
            return Optional.empty();
        }
        JsonNode node = params.get(key);
        if (node == null || node.isNull()) {
            return Optional.empty();
        }
        if (node.isInt() || node.isLong()) {
            return Optional.of(node.asInt());
        }
        try {
            String raw = node.asText();
            if (raw == null || raw.isBlank()) {
                return Optional.empty();
            }
            return Optional.of(Integer.parseInt(raw));
        } catch (Exception ex) {
            return Optional.empty();
        }
    }

    private String normalizeJobType(String jobType) {
        if (jobType == null) {
            return "";
        }
        return jobType.trim().toLowerCase(Locale.ROOT);
    }

    private void validateJobType(String jobType) {
        List<String> supported = List.of("cicids-prepare", "cicids-train-mlp", "cicids-train-bigru", "cicids-eval");
        if (!supported.contains(jobType)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Unsupported jobType: " + jobType);
        }
    }

    private TrainingJobResponse toResponse(TrainingJob job) {
        return new TrainingJobResponse(
            job.getId(),
            job.getJobName(),
            job.getJobType(),
            job.getStatus(),
            job.getRunMessage(),
            job.getExitCode(),
            job.getLogPath(),
            job.getCreateTime(),
            job.getStartTime(),
            job.getEndTime(),
            job.getParamsJson()
        );
    }

    @PreDestroy
    public void shutdownExecutor() {
        executor.shutdownNow();
    }
}
