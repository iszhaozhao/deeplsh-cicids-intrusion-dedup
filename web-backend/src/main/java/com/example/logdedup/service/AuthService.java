package com.example.logdedup.service;

import com.example.logdedup.dto.CurrentUserResponse;
import com.example.logdedup.dto.LoginRequest;
import com.example.logdedup.dto.LoginResponse;
import com.example.logdedup.entity.SysUser;
import com.example.logdedup.repository.SysUserRepository;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

@Service
public class AuthService {

    private final SysUserRepository sysUserRepository;

    public AuthService(SysUserRepository sysUserRepository) {
        this.sysUserRepository = sysUserRepository;
    }

    public LoginResponse login(LoginRequest request) {
        SysUser user = sysUserRepository.findByUsernameAndRole(request.username(), request.role())
            .filter(it -> it.getPassword().equals(request.password()))
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "用户名、密码或角色错误"));
        return new LoginResponse(buildToken(user), user.getUsername(), user.getRealName(), user.getRole());
    }

    public CurrentUserResponse getCurrentUser(String token) {
        SysUser user = resolveToken(token);
        return new CurrentUserResponse(user.getUsername(), user.getRealName(), user.getRole());
    }

    public SysUser resolveToken(String token) {
        if (token == null || token.isBlank()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "缺少登录令牌");
        }
        String raw = token.replace("Bearer ", "");
        String[] parts = raw.split(":");
        if (parts.length != 2) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "无效登录令牌");
        }
        return sysUserRepository.findByUsernameAndRole(parts[0], parts[1])
            .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "用户不存在"));
    }

    private String buildToken(SysUser user) {
        return user.getUsername() + ":" + user.getRole();
    }
}
