package com.example.logdedup.service;

import com.example.logdedup.dto.ResultResponse;
import com.example.logdedup.entity.DedupResult;
import com.example.logdedup.repository.DedupResultRepository;
import java.util.Comparator;
import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

@Service
public class ResultService {

    private final DedupResultRepository dedupResultRepository;

    public ResultService(DedupResultRepository dedupResultRepository) {
        this.dedupResultRepository = dedupResultRepository;
    }

    public List<ResultResponse> listResults(Long taskId) {
        List<DedupResult> results = taskId == null
            ? dedupResultRepository.findAll()
            : dedupResultRepository.findByTaskIdOrderBySimilarityScoreDesc(taskId);
        return results.stream()
            .sorted(Comparator.comparing(DedupResult::getId))
            .map(this::toResponse)
            .toList();
    }

    public ResultResponse getResult(Long id) {
        return dedupResultRepository.findById(id)
            .map(this::toResponse)
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "结果不存在"));
    }

    private ResultResponse toResponse(DedupResult result) {
        return new ResultResponse(
            result.getId(),
            result.getTaskId(),
            result.getLogId(),
            result.getAttackType(),
            result.getHashCode(),
            result.getSimilarityScore(),
            result.getClusterId(),
            result.getIsRedundant(),
            result.getReserveLogId(),
            result.getCandidateSampleId(),
            result.getSourceFile()
        );
    }
}
