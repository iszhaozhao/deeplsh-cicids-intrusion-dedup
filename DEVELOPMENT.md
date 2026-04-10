# 网络入侵检测日志去重系统本地开发环境（macOS）

本仓库包含 3 套独立运行环境：

- Python 模型环境：`code/run.py`
- Web 后端：`web/backend`
- Web 前端：`web/frontend`

## 前置依赖

- Conda
- JDK 17
- Maven 3.9+
- Node.js 18+

## Python 环境

```bash
cd /Users/zhao/domo_codex/deep-locality-sensitive-hashing-main
conda env create -f environment.yml
conda activate deeplsh
```

如需开发期绘图工具：

```bash
conda env create -f environment-dev.yml
conda activate deeplsh-dev
```

### 最小 smoke 流程

```bash
conda run -n deeplsh python code/run.py cicids-prepare --data-repo ./datasets/cicids/raw --output-dir ./datasets/cicids/processed/smoke --max-samples 300 --max-pairs 200
conda run -n deeplsh python code/run.py cicids-train-mlp --data-repo ./datasets/cicids/raw --output-dir ./datasets/cicids/processed/smoke --max-samples 300 --max-pairs 200 --epochs 1
conda run -n deeplsh python code/run.py cicids-train-bigru --data-repo ./datasets/cicids/raw --output-dir ./datasets/cicids/processed/smoke --max-samples 300 --max-pairs 200 --epochs 1
conda run -n deeplsh python code/run.py cicids-eval --output-dir ./datasets/cicids/processed/smoke --results-dir ./artifacts/cicids/results/smoke
```

## Web 后端

```bash
cd web/backend
export DEEPLSH_PYTHON_COMMAND="/opt/miniconda3/bin/conda run -n deeplsh python"
mvn -DskipTests spring-boot:run
```

启动后访问：`http://127.0.0.1:8080`

### H2 控制台

- 地址：`http://127.0.0.1:8080/h2-console`
- JDBC URL：`jdbc:h2:file:./data/logdedup;MODE=MySQL;DB_CLOSE_DELAY=-1;DB_CLOSE_ON_EXIT=FALSE`

## Web 前端

```bash
cd web/frontend
npm install
npm run dev
```

启动后访问：`http://127.0.0.1:5173`

## 推荐启动顺序

1. 先跑 Python smoke，确认 `deeplsh` 环境正常
2. 再启动后端，确认 `DEEPLSH_PYTHON_COMMAND` 可用
3. 最后启动前端，确认 `/api` 代理联通

## 快速验证

```bash
curl -i http://127.0.0.1:8080/h2-console
curl -i http://127.0.0.1:5173/
curl -i http://127.0.0.1:5173/api/auth/me
```
