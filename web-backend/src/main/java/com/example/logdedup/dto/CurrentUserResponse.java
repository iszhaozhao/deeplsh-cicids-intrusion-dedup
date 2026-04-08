package com.example.logdedup.dto;

public record CurrentUserResponse(
    String username,
    String realName,
    String role
) {
}
