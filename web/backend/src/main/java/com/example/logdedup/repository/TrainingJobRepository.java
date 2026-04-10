package com.example.logdedup.repository;

import com.example.logdedup.entity.TrainingJob;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface TrainingJobRepository extends JpaRepository<TrainingJob, Long> {

    List<TrainingJob> findByStatus(String status);
}

