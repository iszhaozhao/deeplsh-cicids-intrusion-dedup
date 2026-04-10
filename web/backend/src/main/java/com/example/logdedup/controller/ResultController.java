package com.example.logdedup.controller;

import com.example.logdedup.dto.ResultResponse;
import com.example.logdedup.service.ResultService;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/results")
public class ResultController {

    private final ResultService resultService;

    public ResultController(ResultService resultService) {
        this.resultService = resultService;
    }

    @GetMapping
    public List<ResultResponse> list(@RequestParam(required = false) Long taskId) {
        return resultService.listResults(taskId);
    }

    @GetMapping("/{id}")
    public ResultResponse get(@PathVariable Long id) {
        return resultService.getResult(id);
    }
}
