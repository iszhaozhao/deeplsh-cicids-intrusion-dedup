package com.example.logdedup.controller;

import com.example.logdedup.dto.LabelStatResponse;
import com.example.logdedup.service.MetaService;
import java.util.List;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/meta")
public class MetaController {

    private final MetaService metaService;

    public MetaController(MetaService metaService) {
        this.metaService = metaService;
    }

    @GetMapping("/labels")
    public List<LabelStatResponse> labels() {
        return metaService.listLabels();
    }
}
