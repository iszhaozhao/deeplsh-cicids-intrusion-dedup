package com.example.logdedup.repository;

import com.example.logdedup.entity.SysUser;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;

public interface SysUserRepository extends JpaRepository<SysUser, Long> {
    Optional<SysUser> findByUsernameAndRole(String username, String role);
    Optional<SysUser> findByUsername(String username);
}
