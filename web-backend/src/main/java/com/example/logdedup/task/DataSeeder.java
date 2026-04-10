package com.example.logdedup.task;

import com.example.logdedup.entity.DedupResult;
import com.example.logdedup.entity.DedupTask;
import com.example.logdedup.entity.SysUser;
import com.example.logdedup.repository.DedupResultRepository;
import com.example.logdedup.repository.DedupTaskRepository;
import com.example.logdedup.repository.SysUserRepository;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

@Component
public class DataSeeder implements CommandLineRunner {

    private final SysUserRepository sysUserRepository;
    private final DedupTaskRepository taskRepository;
    private final DedupResultRepository resultRepository;

    public DataSeeder(
        SysUserRepository sysUserRepository,
        DedupTaskRepository taskRepository,
        DedupResultRepository resultRepository
    ) {
        this.sysUserRepository = sysUserRepository;
        this.taskRepository = taskRepository;
        this.resultRepository = resultRepository;
    }

    @Override
    public void run(String... args) {
        if (sysUserRepository.count() == 0) {
            sysUserRepository.saveAll(List.of(
                buildUser("admin", "admin123", "ADMIN", "系统管理员"),
                buildUser("ops", "ops123", "OPERATOR", "运维人员")
            ));
        }

        if (taskRepository.count() == 0) {
            DedupTask task = new DedupTask();
            task.setTaskName("CICIDS 样例任务");
            task.setSimilarityThreshold(new BigDecimal("0.85"));
            task.setTimeWindow(60);
            task.setReservePolicy("EARLIEST");
            task.setHashBits(32);
            task.setModelType("bigru");
            task.setTotalLogs(10);
            task.setRedundantLogs(6);
            task.setCompressionRate(new BigDecimal("60.00"));
            task.setAvgLatencyMs(new BigDecimal("1.42"));
            task.setStatus("SUCCESS");
            task.setSampleId("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv#0");
            task.setRowIndex(0);
            task.setLabelScope("same");
            task.setTopK(10);
            task.setRunMessage("Seeded demo task");
            task.setCreateTime(LocalDateTime.now().minusDays(1));
            task = taskRepository.save(task);

            for (int i = 0; i < 5; i++) {
                DedupResult result = new DedupResult();
                result.setTaskId(task.getId());
                result.setLogId((long) (i + 1));
                result.setAttackType(i < 3 ? "DDoS" : "PortScan");
                result.setQuerySampleId("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv#0");
                result.setQueryLabel("DDoS");
                result.setCandidateLabel(i < 3 ? "DDoS" : "PortScan");
                result.setHashCode(i % 2 == 0 ? "10101010101010101010101010101010" : "11001100110011001100110011001100");
                result.setSimilarityScore(new BigDecimal("0.9").subtract(new BigDecimal("0.03").multiply(BigDecimal.valueOf(i))));
                result.setHashBucketHits(Math.max(1, 5 - i));
                result.setClusterId("cluster-" + task.getId());
                result.setIsSameLabel(i < 3 ? 1 : 0);
                result.setIsRedundant(i < 3 ? 1 : 0);
                result.setReserveLogId(1L);
                result.setCandidateSampleId("seed-sample-" + i);
                result.setSourceFile("seeded-demo.csv");
                resultRepository.save(result);
            }
        }
    }

    private SysUser buildUser(String username, String password, String role, String realName) {
        SysUser user = new SysUser();
        user.setUsername(username);
        user.setPassword(password);
        user.setRole(role);
        user.setRealName(realName);
        user.setCreateTime(LocalDateTime.now());
        return user;
    }
}
