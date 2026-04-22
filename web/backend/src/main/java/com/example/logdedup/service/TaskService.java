package com.example.logdedup.service;

import com.example.logdedup.dto.PythonQueryRecord;
import com.example.logdedup.dto.TaskCreateRequest;
import com.example.logdedup.dto.TaskResponse;
import com.example.logdedup.entity.DedupResult;
import com.example.logdedup.entity.DedupTask;
import com.example.logdedup.integration.PythonQueryService;
import com.example.logdedup.repository.DedupResultRepository;
import com.example.logdedup.repository.DedupTaskRepository;
import jakarta.transaction.Transactional;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

@Service
public class TaskService {

    private final DedupTaskRepository dedupTaskRepository;
    private final DedupResultRepository dedupResultRepository;
    private final PythonQueryService pythonQueryService;

    public TaskService(
        DedupTaskRepository dedupTaskRepository,
        DedupResultRepository dedupResultRepository,
        PythonQueryService pythonQueryService
    ) {
        this.dedupTaskRepository = dedupTaskRepository;
        this.dedupResultRepository = dedupResultRepository;
        this.pythonQueryService = pythonQueryService;
    }

    public List<TaskResponse> listTasks() {
        return dedupTaskRepository.findAll().stream()
            .sorted((a, b) -> b.getCreateTime().compareTo(a.getCreateTime()))
            .map(this::toResponse)
            .toList();
    }

    public TaskResponse createTask(TaskCreateRequest request) {
        DedupTask task = new DedupTask();
        task.setTaskName(request.taskName());
        task.setModelType(normalizeModelType(request.modelType()));
        task.setSimilarityThreshold(request.similarityThreshold());
        task.setTimeWindow(request.timeWindow());
        task.setReservePolicy(request.reservePolicy());
        task.setHashBits(request.hashBits());
        task.setSampleId(request.sampleId());
        task.setRowIndex(request.rowIndex() == null ? 0 : request.rowIndex());
        task.setLabelScope(request.labelScope() == null ? "same" : request.labelScope());
        task.setTopK(request.topK() == null ? 10 : request.topK());
        task.setStatus("PENDING");
        task.setRunMessage("Task created");
        task.setTotalLogs(0);
        task.setRedundantLogs(0);
        task.setCompressionRate(BigDecimal.ZERO);
        task.setAvgLatencyMs(BigDecimal.ZERO);
        task.setCreateTime(LocalDateTime.now());
        return toResponse(dedupTaskRepository.save(task));
    }

    @Transactional
    public TaskResponse runTask(Long taskId) {
        DedupTask task = dedupTaskRepository.findById(taskId)
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "任务不存在"));

        task.setModelType(normalizeModelType(task.getModelType()));

        task.setStatus("RUNNING");
        task.setRunMessage("Invoking Python query: " + pythonQueryService.describeMode(task));
        dedupTaskRepository.save(task);

        List<DedupResult> existingResults = dedupResultRepository.findByTaskIdOrderBySimilarityScoreDesc(taskId);
        int previousTotalLogs = task.getTotalLogs() == null ? 0 : task.getTotalLogs();
        int previousRedundantLogs = task.getRedundantLogs() == null ? 0 : task.getRedundantLogs();
        BigDecimal previousCompressionRate = task.getCompressionRate() == null ? BigDecimal.ZERO : task.getCompressionRate();
        BigDecimal previousAvgLatencyMs = task.getAvgLatencyMs() == null ? BigDecimal.ZERO : task.getAvgLatencyMs();
        String previousRunMessage = task.getRunMessage();

        try {
            LocalDateTime start = LocalDateTime.now();
            List<PythonQueryRecord> records = pythonQueryService.queryCandidates(task);
            long reserveId = task.getRowIndex() == null ? 1L : task.getRowIndex() + 1L;
            List<DedupResult> newResults = new java.util.ArrayList<>();
            int index = 0;
            for (PythonQueryRecord record : records) {
                DedupResult result = new DedupResult();
                result.setTaskId(taskId);
                result.setLogId((long) (index + 1));
                result.setAttackType(record.candidateLabel());
                result.setQuerySampleId(record.querySampleId());
                result.setQueryLabel(record.queryLabel());
                result.setCandidateLabel(record.candidateLabel());
                result.setCandidateSampleId(record.candidateSampleId());
                result.setHashBucketHits(record.hashBucketHits());
                result.setIsSameLabel(record.isSameLabel());
                result.setClusterId("cluster-" + taskId);
                result.setHashCode(fakeHashCode(record, task.getHashBits() == null ? 32 : task.getHashBits()));
                result.setSimilarityScore(record.embeddingSimilarity().setScale(4, RoundingMode.HALF_UP));
                result.setIsRedundant(record.embeddingSimilarity().compareTo(task.getSimilarityThreshold()) >= 0 ? 1 : 0);
                result.setReserveLogId(reserveId);
                result.setSourceFile(record.sourceFile());
                newResults.add(result);
                index++;
            }

            existingResults.forEach(dedupResultRepository::delete);
            newResults.forEach(dedupResultRepository::save);

            long total = newResults.size();
            long redundant = newResults.stream().filter(item -> Integer.valueOf(1).equals(item.getIsRedundant())).count();
            task.setTotalLogs((int) total);
            task.setRedundantLogs((int) redundant);
            task.setCompressionRate(total == 0
                ? BigDecimal.ZERO
                : BigDecimal.valueOf(redundant * 100.0 / total).setScale(2, RoundingMode.HALF_UP));
            long elapsedMs = Math.max(1L, Duration.between(start, LocalDateTime.now()).toMillis());
            task.setAvgLatencyMs(total == 0
                ? BigDecimal.ZERO
                : BigDecimal.valueOf((double) elapsedMs / total).setScale(2, RoundingMode.HALF_UP));
            task.setStatus("SUCCESS");
            task.setRunMessage("Task finished with " + pythonQueryService.describeMode(task) + " [" + normalizeModelType(task.getModelType()) + "]");
            return toResponse(dedupTaskRepository.save(task));
        } catch (Exception ex) {
            task.setTotalLogs(previousTotalLogs);
            task.setRedundantLogs(previousRedundantLogs);
            task.setCompressionRate(previousCompressionRate);
            task.setAvgLatencyMs(previousAvgLatencyMs);
            task.setStatus("FAILED");
            task.setRunMessage(buildFailureMessage(ex, previousRunMessage));
            dedupTaskRepository.save(task);
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "任务执行失败：" + buildFailureMessage(ex, null), ex);
        }
    }

    public TaskResponse getTask(Long taskId) {
        return dedupTaskRepository.findById(taskId)
            .map(this::toResponse)
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "任务不存在"));
    }

    private TaskResponse toResponse(DedupTask task) {
        return new TaskResponse(
            task.getId(),
            task.getTaskName(),
            normalizeModelType(task.getModelType()),
            task.getSimilarityThreshold(),
            task.getTimeWindow(),
            task.getReservePolicy(),
            task.getHashBits(),
            task.getTotalLogs(),
            task.getRedundantLogs(),
            task.getCompressionRate(),
            task.getAvgLatencyMs(),
            task.getStatus(),
            task.getSampleId(),
            task.getRowIndex(),
            task.getLabelScope(),
            task.getTopK(),
            queryMode(task),
            task.getRunMessage(),
            task.getCreateTime()
        );
    }

    private String fakeHashCode(PythonQueryRecord record, int hashBits) {
        String seed = (record.candidateSampleId() + record.candidateLabel()).replaceAll("[^A-Za-z0-9]", "");
        if (seed.isBlank()) {
            seed = "HASHCODE";
        }
        StringBuilder builder = new StringBuilder(hashBits);
        for (int i = 0; i < hashBits; i++) {
            char ch = seed.charAt(i % seed.length());
            builder.append(((ch + i) % 2 == 0) ? '1' : '0');
        }
        return builder.toString();
    }

    private String normalizeModelType(String modelType) {
        if (modelType == null || modelType.isBlank()) {
            return "bigru";
        }
        return modelType.trim().toLowerCase();
    }

    private String queryMode(DedupTask task) {
        return task.getSampleId() != null && !task.getSampleId().isBlank() ? "sample_id" : "row_index";
    }

    private String buildFailureMessage(Exception ex, String fallbackMessage) {
        String message = ex.getMessage();
        if (message == null || message.isBlank()) {
            message = fallbackMessage;
        }
        if (message == null || message.isBlank()) {
            message = ex.getClass().getSimpleName();
        }
        return message.length() > 240 ? message.substring(0, 240) : message;
    }
}
