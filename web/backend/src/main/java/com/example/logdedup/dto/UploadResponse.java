package com.example.logdedup.dto;

import java.util.List;

public record UploadResponse(
    Long taskId,
    String fileName,
    long totalRows,
    List<String> headers,
    List<List<String>> previewRows,
    String message
) {
}
