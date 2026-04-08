package com.example.logdedup.dto;

public record LoginResponse(
    String token,
    String username,
    String realName,
    String role
) {
}
