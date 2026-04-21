package com.example.logdedup.controller;

import com.example.logdedup.dto.ExperimentMetricResponse;
import com.example.logdedup.dto.ExperimentShowcaseResponse;
import com.example.logdedup.dto.ExperimentSummaryResponse;
import com.example.logdedup.service.ExperimentService;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/experiments")
public class ExperimentController {

    private final ExperimentService experimentService;

    public ExperimentController(ExperimentService experimentService) {
        this.experimentService = experimentService;
    }

    @GetMapping("/metrics")
    public List<ExperimentMetricResponse> metrics() {
        return experimentService.listMetrics();
    }

    @GetMapping("/summary")
    public ExperimentSummaryResponse summary() {
        return experimentService.summary();
    }

    @GetMapping("/showcase")
    public ExperimentShowcaseResponse showcase() {
        return experimentService.showcase();
    }
}
