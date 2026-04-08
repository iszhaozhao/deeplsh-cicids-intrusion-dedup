# Web Backend

Spring Boot demo backend for the intrusion detection log deduplication prototype.

## Features

- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/tasks`
- `GET /api/tasks`
- `POST /api/tasks/{id}/run`
- `GET /api/results`
- `GET /api/results/{id}`
- `POST /api/logs/upload`
- `GET /api/stats/overview`

## Notes

- Development database uses `H2` with MySQL compatibility mode.
- Seed users:
  - `admin / admin123 / ADMIN`
  - `ops / ops123 / OPERATOR`
- Python integration prefers the real `cicids-query` command. If the trained model artifact is missing, the backend falls back to a CSV-driven demo result so the web flow remains usable.

## Run

This project requires Java 17+ and Maven 3.9+.

```bash
mvn spring-boot:run
```
