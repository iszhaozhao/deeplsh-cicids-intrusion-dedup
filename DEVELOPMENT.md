# DeepLSH 本地开发环境（macOS）

本仓库包含 **3 套独立运行环境**：

- **Python（模型/实验）**：`code/run.py`（DeepLSH + CIC-IDS-2017 相关命令）
- **Web 后端**：`web/backend`（Spring Boot 3.x，端口 `8080`）
- **Web 前端**：`web/frontend`（Vue 3 + Vite，端口 `5173`，本地代理 `/api -> 8080`）

---

## 0. 前置依赖

- **Conda**：用于创建/运行 `deeplsh` 环境
- **JDK 17**：运行 Spring Boot 后端
- **Maven 3.9+**：构建/启动后端（本仓库未提供 `mvnw`）
- **Node.js 18+**：运行 Vite 前端（Vite5/Rollup4 要求 node>=18）

---

## 1. Python 环境（DeepLSH / CIC-IDS）

### 创建环境（推荐）

在仓库根目录（`deep-locality-sensitive-hashing-main/`）执行：

```bash
conda env create -f environment.yml
conda activate deeplsh
```

如需 notebook/画图工具：

```bash
conda env create -f environment-dev.yml
conda activate deeplsh-dev
```

> 说明：该仓库对版本较敏感，已固定到 `Python 3.9` 且在 Apple Silicon 上使用 `tensorflow-macos==2.5.0`。

### 最小 smoke test（推荐先跑）

```bash
conda run -n deeplsh python code/run.py list
conda run -n deeplsh python code/run.py lite --measure TraceSim --index-a 0 --index-b 10
conda run -n deeplsh python code/run.py deeplsh --measure TraceSim --n 200 --epochs 1 --batch-size 128
```

---

## 2. Web 后端（Spring Boot）

### 启动

```bash
cd web/backend
mvn -DskipTests spring-boot:run
```

启动成功后：`http://127.0.0.1:8080`

### H2 数据库

开发环境默认使用 **H2 文件库**（数据文件在 `web/backend/data/`），并开启控制台：

- H2 Console：`http://127.0.0.1:8080/h2-console`
- JDBC URL：`jdbc:h2:file:./data/logdedup;MODE=MySQL;DB_CLOSE_DELAY=-1;DB_CLOSE_ON_EXIT=FALSE`

### Python 集成配置（关键）

后端通过配置调用 `code/run.py`。默认配置在：

- `web/backend/src/main/resources/application.yml`

其中 `app.python.command` 支持用环境变量覆盖，默认值为：

- `DEEPLSH_PYTHON_COMMAND="conda run -n deeplsh python"`

如果你的 conda 不在 PATH，或想用指定绝对路径，可这样启动后端：

```bash
export DEEPLSH_PYTHON_COMMAND="/opt/miniconda3/bin/conda run -n deeplsh python"
cd web/backend
mvn -DskipTests spring-boot:run
```

### 常见问题：H2 文件被锁

如果报错类似：

- `Database may be already in use ... logdedup.mv.db`

通常是上一次运行残留了 Java 进程占用文件。处理方式：

```bash
lsof web/backend/data/logdedup.mv.db
kill <pid>
```

---

## 3. Web 前端（Vue3 + Vite）

### 安装依赖并启动

```bash
cd web/frontend
npm install
npm run dev
```

启动成功后：`http://127.0.0.1:5173`

### 代理联调

前端会把 `/api` 代理到后端 `8080`（见 `web/frontend/vite.config.js`）。

---

## 推荐启动顺序（最少踩坑）

1. **先跑 Python CLI smoke test**（确认 `deeplsh` 环境与 `code/run.py` 正常）
2. **再起后端**（确认 8080 正常、且 `DEEPLSH_PYTHON_COMMAND` 可调用 conda 环境）
3. **最后起前端**（确认 5173 正常、`/api` 代理可通）

快速验证命令示例：

```bash
curl -i http://127.0.0.1:8080/h2-console
curl -i http://127.0.0.1:5173/
curl -i http://127.0.0.1:5173/api/auth/me
```
