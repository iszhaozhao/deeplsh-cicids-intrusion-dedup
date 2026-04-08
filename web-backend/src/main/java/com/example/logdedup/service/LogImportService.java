package com.example.logdedup.service;

import com.example.logdedup.dto.UploadResponse;
import com.example.logdedup.entity.IdsLog;
import com.example.logdedup.repository.IdsLogRepository;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

@Service
public class LogImportService {

    private final IdsLogRepository idsLogRepository;

    public LogImportService(IdsLogRepository idsLogRepository) {
        this.idsLogRepository = idsLogRepository;
    }

    public UploadResponse importCsv(Long taskId, MultipartFile file) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(file.getInputStream(), StandardCharsets.UTF_8));
             CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(reader)) {
            List<String> headers = parser.getHeaderNames();
            List<List<String>> previewRows = new ArrayList<>();
            long rowCount = 0;
            for (CSVRecord record : parser) {
                rowCount++;
                List<String> currentRow = new ArrayList<>();
                for (String header : headers) {
                    currentRow.add(record.get(header));
                }
                if (previewRows.size() < 5) {
                    previewRows.add(currentRow);
                }

                IdsLog log = new IdsLog();
                log.setTaskId(taskId);
                log.setEventTime(LocalDateTime.now());
                log.setSrcIp(readMaybe(record, headers, "Src IP", "src_ip"));
                log.setDstIp(readMaybe(record, headers, "Dst IP", "dst_ip"));
                log.setProtocol(readMaybe(record, headers, "Protocol", "protocol"));
                log.setAttackType(readMaybe(record, headers, "Label", "attack_type"));
                log.setLabel(readMaybe(record, headers, "Label", "label"));
                log.setRawText(String.join(" | ", currentRow));
                idsLogRepository.save(log);
            }
            return new UploadResponse(taskId, file.getOriginalFilename(), rowCount, headers, previewRows, "CSV 导入成功");
        }
    }

    private String readMaybe(CSVRecord record, List<String> headers, String... candidates) {
        for (String candidate : candidates) {
            if (headers.contains(candidate)) {
                return record.get(candidate);
            }
        }
        return null;
    }
}
