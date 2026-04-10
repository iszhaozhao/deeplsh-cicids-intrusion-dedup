<template>
  <div class="page">
    <div class="grid">
      <el-card class="panel" shadow="never">
        <template #header>
          <div class="panel-title">论文复现训练（DeepLSH / Stack Traces）</div>
        </template>

        <el-form label-width="110px" class="form">
          <el-form-item label="Measure">
            <el-select v-model="deeplsh.measure" placeholder="选择相似度度量" style="width: 220px">
              <el-option v-for="m in measures" :key="m" :label="m" :value="m" />
            </el-select>
          </el-form-item>

          <el-form-item label="n (stacks)">
            <el-input-number v-model="deeplsh.n" :min="50" :max="1000" />
            <span class="hint">demo 建议 200~400</span>
          </el-form-item>

          <el-form-item label="epochs">
            <el-input-number v-model="deeplsh.epochs" :min="1" :max="50" />
          </el-form-item>

          <el-form-item label="batchSize">
            <el-input-number v-model="deeplsh.batchSize" :min="16" :max="2048" />
          </el-form-item>

          <el-form-item>
            <el-button type="primary" :loading="creating" @click="createAndStartDeeplsh">创建并启动</el-button>
            <el-button @click="presetDeeplshSmoke">Smoke 预设</el-button>
          </el-form-item>
        </el-form>

        <div class="artifacts">
          <div class="artifact-title">预期产物</div>
          <div class="artifact-item">`artifacts/stacktraces/models/model-deep-lsh-&lt;measure&gt;.model`</div>
          <div class="artifact-item">`artifacts/stacktraces/hash_tables/hash_tables_deeplsh_&lt;measure&gt;.pkl`</div>
        </div>
      </el-card>

      <el-card class="panel" shadow="never">
        <template #header>
          <div class="panel-title">本地训练演示（CIC-IDS-2017）</div>
        </template>

        <el-form label-width="110px" class="form">
          <el-form-item label="maxSamples">
            <el-input-number v-model="cicids.maxSamples" :min="200" :max="200000" />
            <span class="hint">demo 建议 300~2000</span>
          </el-form-item>

          <el-form-item label="maxPairs">
            <el-input-number v-model="cicids.maxPairs" :min="200" :max="300000" />
          </el-form-item>

          <el-form-item label="epochs">
            <el-input-number v-model="cicids.epochs" :min="1" :max="50" />
          </el-form-item>

          <el-form-item label="batchSize">
            <el-input-number v-model="cicids.batchSize" :min="16" :max="1024" />
          </el-form-item>

          <el-form-item>
            <el-button type="primary" :loading="creating" @click="createAndStartCicids('cicids-prepare')">1) Prepare</el-button>
            <el-button type="primary" plain :loading="creating" @click="createAndStartCicids('cicids-train-mlp')">2) Train MLP</el-button>
            <el-button type="primary" plain :loading="creating" @click="createAndStartCicids('cicids-train-bigru')">3) Train Bi-GRU</el-button>
            <el-button type="success" plain :loading="creating" @click="createAndStartCicids('cicids-eval')">4) Eval</el-button>
            <el-button @click="presetCicidsSmoke">Smoke 预设</el-button>
          </el-form-item>
        </el-form>

        <div class="artifacts">
          <div class="artifact-title">预期产物</div>
          <div class="artifact-item">`datasets/cicids/processed/full/`（处理后数据）</div>
          <div class="artifact-item">`artifacts/cicids/models/model-deep-lsh-cicids-bigru.model`（论文主模型）</div>
          <div class="artifact-item">`artifacts/cicids/results/full/cicids_comparison_summary.json`（实验对比汇总）</div>
        </div>
      </el-card>
    </div>

    <el-card shadow="never" class="panel">
      <template #header>
        <div class="panel-title">训练任务列表</div>
      </template>

      <el-table :data="jobs" size="small" stripe @row-click="selectJob" height="260">
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column prop="jobName" label="Name" min-width="160" />
        <el-table-column prop="jobType" label="Type" min-width="150" />
        <el-table-column prop="status" label="Status" width="110" />
        <el-table-column prop="runMessage" label="Message" min-width="220" />
        <el-table-column label="操作" width="120">
          <template #default="{ row }">
            <el-button size="small" type="primary" plain @click.stop="startJob(row)" :disabled="row.status === 'RUNNING'">启动</el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="log-area">
        <div class="log-header">
          <div class="log-title">日志（Tail）</div>
          <div class="log-actions">
            <el-input-number v-model="tail" :min="50" :max="400" size="small" />
            <el-button size="small" @click="refreshLogs" :disabled="!selectedJobId">刷新</el-button>
          </div>
        </div>
        <pre class="log-box">{{ logsText }}</pre>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import http from '../api/http'

const measures = ['TraceSim', 'Jaccard', 'Cosine', 'TfIdf', 'Levensh', 'PDM', 'Brodie', 'DURFEX', 'Lerch', 'Moroo']

const creating = ref(false)
const jobs = ref([])
const selectedJobId = ref(null)
const logs = ref([])
const tail = ref(200)
let pollTimer = null

const deeplsh = reactive({
  measure: 'TraceSim',
  n: 200,
  epochs: 1,
  batchSize: 128,
  m: 64,
  b: 16,
  seed: 42,
  lshParamIndex: 2
})

const cicids = reactive({
  dataRepo: './datasets/cicids/raw',
  outputDir: './datasets/cicids/processed/full',
  resultsDir: './artifacts/cicids/results/full',
  maxSamples: 300,
  maxPairs: 200,
  epochs: 1,
  batchSize: 128,
  seed: 42,
  embedDim: 64,
  gruUnits: 64,
  denseDim: 128,
  topK: 10,
  sampleLimit: 50
})

const logsText = computed(() => (logs.value || []).join('\n'))

async function fetchJobs() {
  const res = await http.get('/training/jobs')
  jobs.value = res.data || []
  if (selectedJobId.value) {
    const selected = jobs.value.find((j) => j.id === selectedJobId.value)
    if (selected && selected.status === 'RUNNING') {
      ensurePolling()
    }
  }
}

function selectJob(row) {
  selectedJobId.value = row.id
  refreshLogs()
  if (row.status === 'RUNNING') {
    ensurePolling()
  } else {
    stopPolling()
  }
}

async function startJob(row) {
  try {
    await http.post(`/training/jobs/${row.id}/start`)
    ElMessage.success('已启动训练任务')
    selectedJobId.value = row.id
    await fetchJobs()
    await refreshLogs()
    ensurePolling()
  } catch (e) {
    ElMessage.error(e?.response?.data?.message || '启动失败')
  }
}

async function refreshLogs() {
  if (!selectedJobId.value) return
  try {
    const res = await http.get(`/training/jobs/${selectedJobId.value}/logs`, { params: { tail: tail.value } })
    logs.value = res.data?.lines || []
    const status = res.data?.status
    if (status === 'RUNNING') ensurePolling()
    else stopPolling()
  } catch (e) {
    logs.value = [`ERROR: ${e?.response?.data?.message || e?.message || '无法读取日志'}`]
    stopPolling()
  }
}

function ensurePolling() {
  if (pollTimer) return
  pollTimer = window.setInterval(async () => {
    await fetchJobs()
    await refreshLogs()
  }, 2000)
}

function stopPolling() {
  if (pollTimer) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

function presetDeeplshSmoke() {
  deeplsh.measure = 'TraceSim'
  deeplsh.n = 200
  deeplsh.epochs = 1
  deeplsh.batchSize = 128
}

function presetCicidsSmoke() {
  cicids.maxSamples = 300
  cicids.maxPairs = 200
  cicids.epochs = 1
  cicids.batchSize = 128
}

async function createAndStartDeeplsh() {
  creating.value = true
  try {
    const createRes = await http.post('/training/jobs', {
      jobName: `deeplsh-${deeplsh.measure}-n${deeplsh.n}-e${deeplsh.epochs}`,
      jobType: 'deeplsh-train',
      params: { ...deeplsh }
    })
    const id = createRes.data?.id
    await http.post(`/training/jobs/${id}/start`)
    ElMessage.success('已创建并启动 DeepLSH 训练')
    selectedJobId.value = id
    await fetchJobs()
    await refreshLogs()
    ensurePolling()
  } catch (e) {
    ElMessage.error(e?.response?.data?.message || '创建/启动失败')
  } finally {
    creating.value = false
  }
}

async function createAndStartCicids(jobType) {
  creating.value = true
  try {
    const payload = {
      jobName: `${jobType}-s${cicids.maxSamples}-p${cicids.maxPairs}-e${cicids.epochs}`,
      jobType,
      params: { ...cicids }
    }
    const createRes = await http.post('/training/jobs', payload)
    const id = createRes.data?.id
    await http.post(`/training/jobs/${id}/start`)
    ElMessage.success('已创建并启动训练任务')
    selectedJobId.value = id
    await fetchJobs()
    await refreshLogs()
    ensurePolling()
  } catch (e) {
    ElMessage.error(e?.response?.data?.message || '创建/启动失败')
  } finally {
    creating.value = false
  }
}

onMounted(async () => {
  await fetchJobs()
})

onBeforeUnmount(() => stopPolling())
</script>

<style scoped>
.page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.panel {
  border-radius: 14px;
}

.panel-title {
  font-size: 14px;
  font-weight: 700;
  color: #0f172a;
}

.form :deep(.el-form-item) {
  margin-bottom: 10px;
}

.hint {
  margin-left: 10px;
  color: #64748b;
  font-size: 12px;
}

.artifacts {
  margin-top: 6px;
  padding-top: 10px;
  border-top: 1px dashed #e2e8f0;
  color: #334155;
  font-size: 12px;
}

.artifact-title {
  font-weight: 700;
  margin-bottom: 6px;
}

.artifact-item {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
  margin-bottom: 2px;
}

.log-area {
  margin-top: 14px;
}

.log-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.log-title {
  font-weight: 700;
  color: #0f172a;
}

.log-actions {
  display: flex;
  gap: 10px;
  align-items: center;
}

.log-box {
  margin: 0;
  background: #0b1220;
  color: #e2e8f0;
  padding: 12px;
  border-radius: 12px;
  min-height: 140px;
  max-height: 340px;
  overflow: auto;
  font-size: 12px;
  line-height: 1.4;
}

@media (max-width: 1100px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
