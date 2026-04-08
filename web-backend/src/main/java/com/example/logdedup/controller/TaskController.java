package com.example.logdedup.controller;

import com.example.logdedup.dto.TaskCreateRequest;
import com.example.logdedup.dto.TaskResponse;
import com.example.logdedup.service.TaskService;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/tasks")
public class TaskController {

    private final TaskService taskService;

    public TaskController(TaskService taskService) {
        this.taskService = taskService;
    }

    @GetMapping
    public List<TaskResponse> list() {
        return taskService.listTasks();
    }

    @PostMapping
    public TaskResponse create(@Valid @RequestBody TaskCreateRequest request) {
        return taskService.createTask(request);
    }

    @PostMapping("/{id}/run")
    public TaskResponse run(@PathVariable Long id) {
        return taskService.runTask(id);
    }

    @GetMapping("/{id}")
    public TaskResponse get(@PathVariable Long id) {
        return taskService.getTask(id);
    }
}
