package com.example.logdedup.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import java.math.BigDecimal;

@Entity
@Table(name = "dedup_result")
public class DedupResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "task_id")
    private Long taskId;

    @Column(name = "log_id")
    private Long logId;

    @Column(name = "cluster_id", length = 64)
    private String clusterId;

    @Column(name = "hash_code", length = 128)
    private String hashCode;

    @Column(name = "similarity_score", precision = 5, scale = 4)
    private BigDecimal similarityScore;

    @Column(name = "is_redundant")
    private Integer isRedundant;

    @Column(name = "reserve_log_id")
    private Long reserveLogId;

    @Column(name = "attack_type", length = 50)
    private String attackType;

    @Column(name = "query_sample_id", length = 128)
    private String querySampleId;

    @Column(name = "query_label", length = 64)
    private String queryLabel;

    @Column(name = "candidate_label", length = 64)
    private String candidateLabel;

    @Column(name = "candidate_sample_id", length = 128)
    private String candidateSampleId;

    @Column(name = "hash_bucket_hits")
    private Integer hashBucketHits;

    @Column(name = "is_same_label")
    private Integer isSameLabel;

    @Column(name = "source_file", length = 255)
    private String sourceFile;

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Long getTaskId() {
        return taskId;
    }

    public void setTaskId(Long taskId) {
        this.taskId = taskId;
    }

    public Long getLogId() {
        return logId;
    }

    public void setLogId(Long logId) {
        this.logId = logId;
    }

    public String getClusterId() {
        return clusterId;
    }

    public void setClusterId(String clusterId) {
        this.clusterId = clusterId;
    }

    public String getHashCode() {
        return hashCode;
    }

    public void setHashCode(String hashCode) {
        this.hashCode = hashCode;
    }

    public BigDecimal getSimilarityScore() {
        return similarityScore;
    }

    public void setSimilarityScore(BigDecimal similarityScore) {
        this.similarityScore = similarityScore;
    }

    public Integer getIsRedundant() {
        return isRedundant;
    }

    public void setIsRedundant(Integer isRedundant) {
        this.isRedundant = isRedundant;
    }

    public Long getReserveLogId() {
        return reserveLogId;
    }

    public void setReserveLogId(Long reserveLogId) {
        this.reserveLogId = reserveLogId;
    }

    public String getAttackType() {
        return attackType;
    }

    public void setAttackType(String attackType) {
        this.attackType = attackType;
    }

    public String getQuerySampleId() {
        return querySampleId;
    }

    public void setQuerySampleId(String querySampleId) {
        this.querySampleId = querySampleId;
    }

    public String getQueryLabel() {
        return queryLabel;
    }

    public void setQueryLabel(String queryLabel) {
        this.queryLabel = queryLabel;
    }

    public String getCandidateLabel() {
        return candidateLabel;
    }

    public void setCandidateLabel(String candidateLabel) {
        this.candidateLabel = candidateLabel;
    }

    public String getCandidateSampleId() {
        return candidateSampleId;
    }

    public void setCandidateSampleId(String candidateSampleId) {
        this.candidateSampleId = candidateSampleId;
    }

    public Integer getHashBucketHits() {
        return hashBucketHits;
    }

    public void setHashBucketHits(Integer hashBucketHits) {
        this.hashBucketHits = hashBucketHits;
    }

    public Integer getIsSameLabel() {
        return isSameLabel;
    }

    public void setIsSameLabel(Integer isSameLabel) {
        this.isSameLabel = isSameLabel;
    }

    public String getSourceFile() {
        return sourceFile;
    }

    public void setSourceFile(String sourceFile) {
        this.sourceFile = sourceFile;
    }
}
