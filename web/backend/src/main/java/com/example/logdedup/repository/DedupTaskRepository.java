package com.example.logdedup.repository;

import com.example.logdedup.entity.DedupTask;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DedupTaskRepository extends JpaRepository<DedupTask, Long> {
}
