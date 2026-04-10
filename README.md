# 基于深度哈希的网络入侵检测日志去重系统

## 项目简介

本仓库是一个面向本科毕业设计的网络入侵检测日志去重系统原型，研究对象为 `CIC-IDS-2017` 网络流量日志。项目围绕“近似重复日志识别”这一问题，构建了从数据预处理、深度哈希训练、近邻检索到 Web 展示的一体化闭环。

当前仓库只保留与该毕设主线相关的内容：

- Python 训练与评估管线
- Spring Boot 后端
- Vue 3 前端原型
- 毕设答辩与使用文档

旧版 notebook 研究内容已从主版本移除，当前主版本仅保留网络入侵检测日志去重系统所需内容。

## 目录结构

```text
.
├── code/                    # CLI 入口
├── python/src/deeplsh/      # Python 算法实现
├── web/backend/             # Spring Boot 后端
├── web/frontend/            # Vue 3 前端
├── docs/                    # 使用说明、答辩材料
├── environment.yml          # Python 运行环境
└── environment-dev.yml      # 开发/绘图补充环境
```

## 技术栈

- Python 3.9
- TensorFlow / tensorflow-macos
- Spring Boot 3.x
- Vue 3 + Vite + Element Plus
- H2（开发默认，可扩展 MySQL）

## Python 模型流程

推荐使用 `conda` 环境：

```bash
conda env create -f environment.yml
conda activate deeplsh
```

### 1. 准备 CIC-IDS 数据

```bash
python code/run.py cicids-prepare \
  --data-repo ./datasets/cicids/raw \
  --output-dir ./datasets/cicids/processed/full \
  --max-samples 12000 \
  --max-pairs 20000
```

### 2. 训练 MLP 基线

```bash
python code/run.py cicids-train-mlp \
  --data-repo ./datasets/cicids/raw \
  --output-dir ./datasets/cicids/processed/full \
  --epochs 10 \
  --batch-size 256
```

### 3. 训练 Bi-GRU 主模型

```bash
python code/run.py cicids-train-bigru \
  --data-repo ./datasets/cicids/raw \
  --output-dir ./datasets/cicids/processed/full \
  --epochs 10 \
  --batch-size 128
```

### 4. 评估与对比

```bash
python code/run.py cicids-eval \
  --output-dir ./datasets/cicids/processed/full \
  --results-dir ./artifacts/cicids/results/full
```

### 5. 查询近似重复结果

```bash
python code/run.py cicids-query --model-type bigru --row-index 0 --top-k 10
python code/run.py cicids-query --model-type mlp --row-index 0 --label-scope all --top-k 10
```

### Smoke 演示参数

如果只是本地演示或答辩前联调，建议先跑小规模 smoke：

```bash
python code/run.py cicids-prepare --data-repo ./datasets/cicids/raw --output-dir ./datasets/cicids/processed/smoke --max-samples 300 --max-pairs 200
python code/run.py cicids-train-mlp --data-repo ./datasets/cicids/raw --output-dir ./datasets/cicids/processed/smoke --max-samples 300 --max-pairs 200 --epochs 1
python code/run.py cicids-train-bigru --data-repo ./datasets/cicids/raw --output-dir ./datasets/cicids/processed/smoke --max-samples 300 --max-pairs 200 --epochs 1
python code/run.py cicids-eval --output-dir ./datasets/cicids/processed/smoke --results-dir ./artifacts/cicids/results/smoke
```

## Web 原型启动

### 1. 启动后端

```bash
cd web/backend
export DEEPLSH_PYTHON_COMMAND="/opt/miniconda3/bin/conda run -n deeplsh python"
mvn -DskipTests spring-boot:run
```

后端地址：`http://127.0.0.1:8080`

### 2. 启动前端

```bash
cd web/frontend
npm install
npm run dev
```

前端地址：`http://127.0.0.1:5173`

## Web 演示流程

1. 登录系统
2. 在“参数配置”中创建任务
3. 在“任务执行”中启动任务
4. 在“结果展示”中查看冗余标识与聚类详情
5. 在“统计分析”中展示压缩率、攻击类型分布与时延
6. 如需演示训练过程，进入“训练演示”页执行 `Prepare -> Train MLP -> Train Bi-GRU -> Eval`

## 重要产物

- 处理后数据：`datasets/cicids/processed/full/`
- 模型文件：`artifacts/cicids/models/`
- 哈希表：`artifacts/cicids/hash_tables/`
- 评估结果：`artifacts/cicids/results/full/`

## 文档入口

- [docs/WEB_USAGE_GUIDE.md](/Users/zhao/domo_codex/deep-locality-sensitive-hashing-main/docs/WEB_USAGE_GUIDE.md)
- [docs/答辩演示脚本.md](/Users/zhao/domo_codex/deep-locality-sensitive-hashing-main/docs/答辩演示脚本.md)
- [docs/答辩PPT主线.md](/Users/zhao/domo_codex/deep-locality-sensitive-hashing-main/docs/答辩PPT主线.md)
- [DEVELOPMENT.md](/Users/zhao/domo_codex/deep-locality-sensitive-hashing-main/DEVELOPMENT.md)
