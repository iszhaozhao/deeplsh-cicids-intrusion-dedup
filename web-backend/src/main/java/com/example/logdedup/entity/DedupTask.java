package com.example.logdedup.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.math.BigDecimal;
import java.time.LocalDateTime;

@Entity
@Table(name = "dedup_task")
public class DedupTask {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "task_name", nullable = false, length = 50)
    private String taskName;

    @Column(name = "similarity_threshold", precision = 4, scale = 2)
    private BigDecimal similarityThreshold;

    @Column(name = "time_window")
    private Integer timeWindow;

    @Column(name = "reserve_policy", length = 20)
    private String reservePolicy;

    @Column(name = "hash_bits")
    private Integer hashBits;

    @Column(name = "model_type", length = 16)
    private String modelType;

    @Column(name = "total_logs")
    private Integer totalLogs;

    @Column(name = "redundant_logs")
    private Integer redundantLogs;

    @Column(name = "compression_rate", precision = 5, scale = 2)
    private BigDecimal compressionRate;

    @Column(name = "avg_latency_ms", precision = 8, scale = 2)
    private BigDecimal avgLatencyMs;

    @Column(length = 20)
    private String status;

    @Column(name = "sample_id", length = 128)
    private String sampleId;

    @Column(name = "row_index")
    private Integer rowIndex;

    @Column(name = "label_scope", length = 10)
    private String labelScope;

    @Column(name = "top_k")
    private Integer topK;

    @Column(name = "run_message", length = 500)
    private String runMessage;

    @Column(name = "create_time", nullable = false)
    private LocalDateTime createTime;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTaskName() {
        return taskName;
    }

    public void setTaskName(String taskName) {
        this.taskName = taskName;
    }

    public BigDecimal getSimilarityThreshold() {
        return similarityThreshold;
    }

    public void setSimilarityThreshold(BigDecimal similarityThreshold) {
        this.similarityThreshold = similarityThreshold;
    }

    public Integer getTimeWindow() {
        return timeWindow;
    }

    public void setTimeWindow(Integer timeWindow) {
        this.timeWindow = timeWindow;
    }

    public String getReservePolicy() {
        return reservePolicy;
    }

    public void setReservePolicy(String reservePolicy) {
        this.reservePolicy = reservePolicy;
    }

    public Integer getHashBits() {
        return hashBits;
    }

    public void setHashBits(Integer hashBits) {
        this.hashBits = hashBits;
    }

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public Integer getTotalLogs() {
        return totalLogs;
    }

    public void setTotalLogs(Integer totalLogs) {
        this.totalLogs = totalLogs;
    }

    public Integer getRedundantLogs() {
        return redundantLogs;
    }

    public void setRedundantLogs(Integer redundantLogs) {
        this.redundantLogs = redundantLogs;
    }

    public BigDecimal getCompressionRate() {
        return compressionRate;
    }

    public void setCompressionRate(BigDecimal compressionRate) {
        this.compressionRate = compressionRate;
    }

    public BigDecimal getAvgLatencyMs() {
        return avgLatencyMs;
    }

    public void setAvgLatencyMs(BigDecimal avgLatencyMs) {
        this.avgLatencyMs = avgLatencyMs;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getSampleId() {
        return sampleId;
    }

    public void setSampleId(String sampleId) {
        this.sampleId = sampleId;
    }

    public Integer getRowIndex() {
        return rowIndex;
    }

    public void setRowIndex(Integer rowIndex) {
        this.rowIndex = rowIndex;
    }

    public String getLabelScope() {
        return labelScope;
    }

    public void setLabelScope(String labelScope) {
        this.labelScope = labelScope;
    }

    public Integer getTopK() {
        return topK;
    }

    public void setTopK(Integer topK) {
        this.topK = topK;
    }

    public String getRunMessage() {
        return runMessage;
    }

    public void setRunMessage(String runMessage) {
        this.runMessage = runMessage;
    }

    public LocalDateTime getCreateTime() {
        return createTime;
    }

    public void setCreateTime(LocalDateTime createTime) {
        this.createTime = createTime;
    }
}
