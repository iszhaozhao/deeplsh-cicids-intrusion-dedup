package com.example.logdedup.controller;

import com.example.logdedup.dto.UploadResponse;
import com.example.logdedup.service.LogImportService;
import java.io.IOException;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/logs")
public class LogController {

    private final LogImportService logImportService;

    public LogController(LogImportService logImportService) {
        this.logImportService = logImportService;
    }

    @PostMapping("/upload")
    public UploadResponse upload(
        @RequestParam("taskId") Long taskId,
        @RequestParam("file") MultipartFile file
    ) throws IOException {
        return logImportService.importCsv(taskId, file);
    }
}
