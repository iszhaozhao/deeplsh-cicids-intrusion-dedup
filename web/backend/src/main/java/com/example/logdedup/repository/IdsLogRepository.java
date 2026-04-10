package com.example.logdedup.repository;

import com.example.logdedup.entity.IdsLog;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface IdsLogRepository extends JpaRepository<IdsLog, Long> {
    List<IdsLog> findByTaskIdOrderByIdAsc(Long taskId);
    long countByTaskId(Long taskId);
}
