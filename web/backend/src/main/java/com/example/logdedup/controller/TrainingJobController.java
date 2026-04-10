package com.example.logdedup.controller;

import com.example.logdedup.dto.TrainingJobCreateRequest;
import com.example.logdedup.dto.TrainingJobLogResponse;
import com.example.logdedup.dto.TrainingJobResponse;
import com.example.logdedup.service.TrainingJobService;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/training/jobs")
public class TrainingJobController {

    private final TrainingJobService trainingJobService;

    public TrainingJobController(TrainingJobService trainingJobService) {
        this.trainingJobService = trainingJobService;
    }

    @GetMapping
    public List<TrainingJobResponse> list() {
        return trainingJobService.listJobs();
    }

    @PostMapping
    public TrainingJobResponse create(@Valid @RequestBody TrainingJobCreateRequest request) {
        return trainingJobService.createJob(request);
    }

    @PostMapping("/{id}/start")
    public TrainingJobResponse start(@PathVariable Long id) {
        return trainingJobService.startJob(id);
    }

    @GetMapping("/{id}")
    public TrainingJobResponse get(@PathVariable Long id) {
        return trainingJobService.getJob(id);
    }

    @GetMapping("/{id}/logs")
    public TrainingJobLogResponse logs(@PathVariable Long id, @RequestParam(name = "tail", required = false) Integer tail) {
        return trainingJobService.tailLogs(id, tail);
    }
}

