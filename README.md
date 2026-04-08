# DeepLSH: Deep Locality-Sensitive Hash Learning for Fast and Efficient Near-Duplicate Crash Report Detection

## Overview
Automatic crash bucketing is a critical step in the software development process to efficiently analyze and triage bug reports. In this work, we aim at detecting for a crash report its candidate near-duplicates (i.e., similar crashes that are likely to be induced by the same software bug) in a large database of historical crashes and given any similarity measure dedicated to compare between stack traces. To this end, we propose **DeepLSH** a deep Siamese hash coding neural network based on Locality-Sensitive Hashing (LSH) property in order to provide binary hash codes aiming to locate the most similar stack traces into hash buckets. **DeepLSH** have been conducted on a large stack trace dataset and performed on state-of-the-art similarity measures proposed to tackle the crash deduplication problem:
- Jaccard coefficient [[Ref](https://en.wikipedia.org/wiki/Jaccard_index)]
- Cosine similarity [[Ref](https://en.wikipedia.org/wiki/Sine_and_cosine)]
- Lucene TF-IDF [[Ref](https://lucene.apache.org/core/7_6_0/core/org/apache/lucene/search/similarities/TFIDFSimilarity.html)]
- Edit distance [[Ref](https://en.wikipedia.org/wiki/Edit_distance)]
- Brodie et al. [[Paper](https://www.cs.drexel.edu/~spiros/teaching/CS576/papers/Brodie_ICAC05.pdf)]
- PDM-Rebucket [[Paper](https://www.researchgate.net/publication/254041628_ReBucket_A_method_for_clustering_duplicate_crash_reports_based_on_call_stack_similarity)]
- DURFEX [[Paper](https://users.encs.concordia.ca/~abdelw/papers/QRS17-Durfex.pdf)]
- Lerch and Mezini [[Paper](https://files.inria.fr/sachaproject/htdocs//lerch2013.pdf)]
- Moroo et al. [[Paper](http://ksiresearch.org/seke/seke17paper/seke17paper_135.pdf)]
- TraceSIM [[Paper](https://arxiv.org/pdf/2009.12590.pdf)]

## Contributions

Our contribution is three-fold. 
- Aiming to overcome the problem of deriving LSH functions for stack-trace similarity measures, we propose a generic approach dubbed DeepLSH that learns and provides a family of binary hash functions that perfectly approximate the locality-sensitive property to retrieve efficiently and rapidly near-duplicate stack traces. 

![lsh](code/Images/lshPhases.png)

- Technically, we design a deep Siamese neural network architecture to perform end-to-end hashing with an original objective loss function based on the locality-sensitive property preserving with appropriate regularizations to cope with the binarization problem of optimizing non-smooth loss functions. 
- We demonstrate through our experimental study the effectiveness and scalability of DeepLSH to yield near-duplicate crash reports under a dozen of similarity metrics. We successfully compare to standard LSH techniques (MinHash and SimHash), and the most relevant deep hashing baselineon a large real-world dataset that we make available.

![contrib](code/Images/Images-paper/DeepLSH%20model.png)

## How to use this code?
我推荐使用的是python3.9
1. Clone this repository 
2. Install the required python packages: ```pip install -r ./code/requirements.txt ```
3. Run without Jupyter (recommended for local CLI usage):
    
    3.1. Create and activate a virtual environment (Windows PowerShell example):
    ```
    py -3.9 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r .\code\requirements.txt
    ```

    3.2. List available similarity measures (from `data/similarity-measures-pairs.csv`):
    ```
    python .\code\run.py list
    ```

    3.3. Lightweight (lite) run (fast): query one similarity value from the precomputed pairs file.
    This is the fastest way to "run the project" locally and choose a measure.
    ```
    python .\code\run.py lite --measure TraceSim --index-a 0 --index-b 10
    python .\code\run.py lite --measure Jaccard --index-a 0 --index-b 10
    python .\code\run.py lite --measure Brodie --index-a 0 --index-b 10
    python .\code\run.py lite --measure DURFEX --index-a 0 --index-b 10
    python .\code\run.py lite --measure TfIdf --index-a 0 --index-b 10
    ```
    Note: the provided `similarity-measures-pairs.csv` corresponds to 1000 stacks in this repo.

    3.4. Full run (DeepLSH training + LSH hash tables): train DeepLSH for a selected measure and build hash tables.
    ```
    python .\code\run.py deeplsh --measure TraceSim
    python .\code\run.py deeplsh --measure Jaccard
    python .\code\run.py deeplsh --measure Cosine
    python .\code\run.py deeplsh --measure TfIdf
    python .\code\run.py deeplsh --measure Levensh
    python .\code\run.py deeplsh --measure PDM
    python .\code\run.py deeplsh --measure Brodie
    python .\code\run.py deeplsh --measure DURFEX
    python .\code\run.py deeplsh --measure Lerch
    python .\code\run.py deeplsh --measure Moroo
    ```
    Outputs:
    - Models are saved to `code/Models/` as `model-deep-lsh-<measure>.model`
    - Hash tables are saved to `code/Hash-Tables/` as `hash_tables_deeplsh_<measure>.pkl`

    3.5. Faster smoke test for DeepLSH (recommended first run):
    ```
    python .\code\run.py deeplsh --measure TraceSim --n 200 --epochs 1 --batch-size 128
    ```

 4. (Optional) Run the notebooks in `code/notebooks/` if you prefer the original experimental setup.

## CIC-IDS-2017 network flow deduplication
This repo now supports two CIC-IDS-2017 experiment paths in `MachineLearningCVE/`:

- `flow-mlp baseline`: numeric flow features + MLP + DeepLSH
- `bigru-deeplsh`: tokenized log-style sequences + Bi-GRU + DeepLSH

1. Prepare processed flow and sequence data:
```
python code/run.py cicids-prepare --data-repo ./MachineLearningCVE --max-samples 12000 --max-pairs 20000
python code/run.py cicids-prepare-flow --data-repo ./MachineLearningCVE --max-samples 12000 --max-pairs 20000
python code/run.py cicids-prepare-seq --data-repo ./MachineLearningCVE --max-samples 12000 --max-pairs 20000
```

2. Inspect label distribution:
```
python code/run.py cicids-list-labels
python code/run.py cicids-list-labels --from-raw --data-repo ./MachineLearningCVE
```

3. Train the CIC-IDS MLP baseline and Bi-GRU paper model:
```
python code/run.py cicids-train --data-repo ./MachineLearningCVE --epochs 10 --batch-size 256
python code/run.py cicids-train-mlp --data-repo ./MachineLearningCVE --epochs 10 --batch-size 256
python code/run.py cicids-train-bigru --data-repo ./MachineLearningCVE --epochs 10 --batch-size 128
```

4. Query near-duplicate flows:
```
python code/run.py cicids-query --model-type bigru --row-index 0 --label-scope same --top-k 10
python code/run.py cicids-query --model-type mlp --sample-id Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv#0 --label-scope all --top-k 10
```

5. Run paper-style evaluation:
```
python code/run.py cicids-eval --output-dir ./data/cic_ids2017_processed
```

Artifacts:
- Processed flow data is written to `data/cic_ids2017_processed/`
- Model artifacts are written to `code/Models/`
- LSH hash tables are written to `code/Hash-Tables/`
- Evaluation results are written to `results/`

## Conda (recommended if you don't use Docker)
On macOS Apple Silicon (arm64), use the native `tensorflow-macos` build via `environment.yml`.

1. Create the environment:
```
conda env create -f environment.yml
conda activate deeplsh
```

2. Smoke test:
```
python code/run.py list
python code/run.py lite --measure TraceSim --index-a 0 --index-b 10
python code/run.py deeplsh --measure TraceSim --n 200 --epochs 1 --batch-size 128
```

If you prefer not to activate the shell environment, the same commands can be run with `conda run`:
```
/opt/miniconda3/bin/conda run -n deeplsh python code/run.py list
/opt/miniconda3/bin/conda run -n deeplsh python code/run.py cicids-prepare --data-repo ./MachineLearningCVE --max-samples 300 --max-pairs 200
/opt/miniconda3/bin/conda run -n deeplsh python code/run.py cicids-train-mlp --data-repo ./MachineLearningCVE --max-samples 300 --max-pairs 200 --epochs 1
/opt/miniconda3/bin/conda run -n deeplsh python code/run.py cicids-train-bigru --data-repo ./MachineLearningCVE --max-samples 300 --max-pairs 200 --epochs 1
/opt/miniconda3/bin/conda run -n deeplsh python code/run.py cicids-eval --output-dir ./data/cic_ids2017_processed
/opt/miniconda3/bin/conda run -n deeplsh python code/run.py cicids-query --model-type bigru --row-index 0 --top-k 5
```

## Environment layout
This project currently uses three separate runtime environments. Only the model pipeline uses a dedicated Python virtual environment.

1. Python model environment
- Purpose: CIC-IDS-2017 preprocessing, training, evaluation, and near-duplicate query.
- Type: `conda` virtual environment.
- Name: `deeplsh`.
- Definition: `environment.yml`.
- Recommended usage:
```
conda run -n deeplsh python code/run.py cicids-prepare --data-repo ./MachineLearningCVE
conda run -n deeplsh python code/run.py cicids-train-bigru --data-repo ./MachineLearningCVE
```

2. Web frontend environment
- Purpose: Vue-based visualization and demo UI.
- Type: project-local Node.js environment.
- Definition: `web-frontend/package.json`.
- Typical commands:
```
cd web-frontend
npm install
npm run dev
```

3. Web backend environment
- Purpose: Spring Boot API layer, task orchestration, and Python CLI integration.
- Type: local Java/Maven environment.
- Definition: `web-backend/pom.xml`.
- Required runtime: `JDK 17`.
- Typical commands:
```
cd web-backend
./mvnw spring-boot:run
```

Notes and caveats:
- Do not run the model code from the conda `base` environment. Use `deeplsh`.
- Prefer `environment.yml` over `code/requirements.txt`. The latter is mainly a legacy dependency list from the original project.
- The Python stack is version-sensitive. This repo is currently aligned to `Python 3.9` and `tensorflow-macos==2.5.0`.
- The backend configuration in `web-backend/src/main/resources/application.yml` uses absolute local paths for the Python workdir, script, model artifact, and flow CSV. Update those paths if you move the repo or switch machines.
- The frontend and backend are not managed by conda. Installing the Python environment does not install Node.js or Java dependencies.
- The backend uses a local H2 file database (`jdbc:h2:file:./data/logdedup`), so database files are created relative to the backend working directory.

## Docker (recommended on macOS Apple Silicon)
This repo pins `tensorflow==2.5.0`, which is easiest to run in a Linux container on Apple Silicon.

1. Build (force `linux/amd64`):
```
docker buildx build --platform linux/amd64 -t deeplsh:py39 .
```

2. Run examples:
```
docker run --rm -it --platform linux/amd64 -v "$PWD":/app -w /app deeplsh:py39 python code/run.py list
docker run --rm -it --platform linux/amd64 -v "$PWD":/app -w /app deeplsh:py39 python code/run.py lite --measure TraceSim --index-a 0 --index-b 10
docker run --rm -it --platform linux/amd64 -v "$PWD":/app -w /app deeplsh:py39 python code/run.py deeplsh --measure TraceSim --n 200 --epochs 1 --batch-size 128
```
