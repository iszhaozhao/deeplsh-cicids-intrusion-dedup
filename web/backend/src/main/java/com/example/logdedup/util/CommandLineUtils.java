package com.example.logdedup.util;

import java.util.ArrayList;
import java.util.List;

public final class CommandLineUtils {

    private CommandLineUtils() {
    }

    public static List<String> splitCommand(String command) {
        if (command == null || command.isBlank()) {
            return List.of();
        }
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inSingle = false;
        boolean inDouble = false;
        boolean escaping = false;

        for (int i = 0; i < command.length(); i++) {
            char ch = command.charAt(i);

            if (escaping) {
                current.append(ch);
                escaping = false;
                continue;
            }

            if (ch == '\\') {
                escaping = true;
                continue;
            }

            if (ch == '\'' && !inDouble) {
                inSingle = !inSingle;
                continue;
            }

            if (ch == '"' && !inSingle) {
                inDouble = !inDouble;
                continue;
            }

            if (!inSingle && !inDouble && Character.isWhitespace(ch)) {
                if (current.length() > 0) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }
                continue;
            }

            current.append(ch);
        }

        if (current.length() > 0) {
            tokens.add(current.toString());
        }
        return tokens;
    }
}

