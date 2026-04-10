package com.example.logdedup.repository;

import com.example.logdedup.entity.DedupResult;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DedupResultRepository extends JpaRepository<DedupResult, Long> {
    List<DedupResult> findByTaskIdOrderBySimilarityScoreDesc(Long taskId);
    long countByTaskId(Long taskId);
    long countByTaskIdAndIsRedundant(Long taskId, Integer isRedundant);
}
