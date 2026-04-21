<template>
  <div class="page-stack">
    <el-card shadow="never" class="showcase-hero">
      <div class="showcase-hero__content">
        <div class="showcase-hero__text">
          <div class="showcase-kicker">CIC-IDS-2017 Full 正式成果</div>
          <h2>基于深度哈希的网络入侵检测日志去重成果展示</h2>
          <p>
            {{ showcase.conclusion?.summary || '当前页面用于集中展示 full 口径实验结论、模型对比结果和系统闭环产物。' }}
          </p>
          <div class="showcase-tags">
            <el-tag size="large" effect="dark" type="success">{{ showcase.datasetName || 'CIC-IDS-2017' }}</el-tag>
            <el-tag size="large" type="info">flows {{ formatCount(showcase.flowCount) }}</el-tag>
            <el-tag size="large" type="warning">pairs {{ formatCount(showcase.pairCount) }}</el-tag>
            <el-tag size="large">{{ showcase.bestModel?.displayName || '暂无最佳模型' }}</el-tag>
          </div>
        </div>

        <div class="showcase-hero__callout">
          <div class="callout-label">论文主结论</div>
          <div class="callout-title">{{ showcase.conclusion?.headline || '尚未生成结论' }}</div>
          <div class="callout-meta">
            <span>最佳 F1：{{ formatRatio(showcase.bestModel?.f1) }}</span>
            <span>Recall：{{ formatRatio(showcase.bestModel?.recall) }}</span>
            <span>Compression：{{ formatRatio(showcase.bestModel?.compressionRate) }}</span>
          </div>
        </div>
      </div>
    </el-card>

    <el-card shadow="never" v-if="loading">
      <el-skeleton :rows="8" animated />
    </el-card>

    <el-card shadow="never" v-else-if="errorText">
      <el-result icon="error" title="成果数据加载失败" :sub-title="errorText" />
    </el-card>

    <el-card shadow="never" v-else-if="!showcase.hasData">
      <el-empty description="未检测到 full 口径实验结果，请先确认 artifacts/cicids/results/full 中的指标文件。" />
    </el-card>

    <template v-else>
      <div class="metric-grid">
        <MetricCard label="最佳 F1" :value="formatRatio(showcase.bestModel?.f1)" hint="按 F1 -> Recall -> Precision 排序得出" />
        <MetricCard label="最佳 Recall" :value="formatRatio(showcase.bestModel?.recall)" hint="更适合入侵检测场景的召回优先目标" />
        <MetricCard label="压缩率" :value="formatRatio(showcase.bestModel?.compressionRate)" hint="反映日志去重后的空间压缩效果" />
        <MetricCard label="平均查询时延" :value="formatLatency(showcase.bestModel?.avgQueryLatencyMs)" hint="Top-K 查询平均延迟" />
      </div>

      <div class="panel-grid">
        <el-card shadow="never">
          <template #header>四模型 Precision / Recall / F1 对比</template>
          <SimpleChart :option="prfOption" />
        </el-card>
        <el-card shadow="never">
          <template #header>压缩率与时延对比</template>
          <SimpleChart :option="compressionLatencyOption" />
        </el-card>
      </div>

      <div class="panel-grid">
        <el-card shadow="never">
          <template #header>结论解读</template>
          <div class="insight-stack">
            <div class="insight-block">
              <div class="insight-title">为什么选择 Bi-GRU + DeepLSH</div>
              <div class="insight-body">
                Bi-GRU 能编码流量序列上下文，再通过 DeepLSH 映射到便于近似检索的哈希空间，因此在 full 口径下取得了当前最佳 F1 与更高 Recall。
              </div>
            </div>
            <div class="insight-block">
              <div class="insight-title">相比 MLP baseline 的提升</div>
              <div class="insight-body">
                <div>F1 提升：{{ formatDelta(showcase.conclusion?.baselineDelta?.deltaF1) }}</div>
                <div>Recall 提升：{{ formatDelta(showcase.conclusion?.baselineDelta?.deltaRecall) }}</div>
                <div>Compression 提升：{{ formatDelta(showcase.conclusion?.baselineDelta?.deltaCompressionRate) }}</div>
                <div>Latency 变化：{{ formatLatencyDelta(showcase.conclusion?.baselineDelta?.deltaLatencyMs) }}</div>
              </div>
            </div>
            <div class="insight-block">
              <div class="insight-title">相比 MD5 / SimHash 的意义</div>
              <div class="insight-body">
                MD5 只能覆盖完全重复日志，SimHash 能处理一定相似性但仍依赖浅层签名；深度表示学习方案能更稳定地识别近似重复与同类攻击模式。
              </div>
            </div>
            <div class="insight-block emphasis">
              <div class="insight-title">推荐结论</div>
              <div class="insight-body">{{ showcase.conclusion?.recommendation }}</div>
            </div>
          </div>
        </el-card>

        <el-card shadow="never">
          <template #header>训练与评估口径</template>
          <div class="detail-stack">
            <div><strong>数据集：</strong>{{ showcase.datasetName }}</div>
            <div><strong>样本规模：</strong>{{ formatCount(showcase.flowCount) }} flows / {{ formatCount(showcase.pairCount) }} pairs</div>
            <div><strong>评估参数：</strong>Top-K = {{ showcase.topK ?? '-' }}, Sample Limit = {{ showcase.sampleLimit ?? '-' }}</div>
            <div><strong>最佳模型：</strong>{{ showcase.bestModel?.displayName || '-' }}</div>
            <div><strong>正式口径：</strong>使用仓库内已固定的 full 训练与评估结果作为论文和答辩展示口径。</div>
          </div>

          <el-table :data="showcase.models" size="small" class="showcase-table">
            <el-table-column prop="displayName" label="模型" min-width="220" />
            <el-table-column label="F1" width="110">
              <template #default="{ row }">{{ formatRatio(row.f1) }}</template>
            </el-table-column>
            <el-table-column label="Recall" width="110">
              <template #default="{ row }">{{ formatRatio(row.recall) }}</template>
            </el-table-column>
            <el-table-column label="Compression" width="140">
              <template #default="{ row }">{{ formatRatio(row.compressionRate) }}</template>
            </el-table-column>
            <el-table-column label="Latency" width="130">
              <template #default="{ row }">{{ formatLatency(row.avgQueryLatencyMs) }}</template>
            </el-table-column>
            <el-table-column label="结论" width="100">
              <template #default="{ row }">
                <el-tag :type="row.isBest ? 'success' : 'info'">{{ row.isBest ? '最佳' : '对照' }}</el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </div>

      <el-card shadow="never">
        <template #header>落地产物与演示闭环</template>
        <div class="artifact-grid">
          <div class="artifact-panel">
            <div class="artifact-title">命令口径</div>
            <div class="artifact-code">python code/run.py cicids-train-mlp --output-dir ./datasets/cicids/processed/full --epochs 1</div>
            <div class="artifact-code">python code/run.py cicids-train-bigru --output-dir ./datasets/cicids/processed/full --epochs 1</div>
            <div class="artifact-code">python code/run.py cicids-eval --output-dir ./datasets/cicids/processed/full --results-dir ./artifacts/cicids/results/full</div>
          </div>

          <div class="artifact-panel">
            <div class="artifact-title">结果文件</div>
            <div class="artifact-code">{{ showcase.artifacts?.summaryJson || '-' }}</div>
            <div class="artifact-code">{{ showcase.artifacts?.baselineMetricsCsv || '-' }}</div>
            <div class="artifact-code">{{ showcase.artifacts?.bigruMetricsCsv || '-' }}</div>
          </div>

          <div class="artifact-panel">
            <div class="artifact-title">系统闭环</div>
            <div class="artifact-text">数据预处理 -> 深度哈希训练 -> Top-K 近似检索 -> Web 可视化展示</div>
            <div class="artifact-text">推荐演示顺序：登录 -> 成果展示 -> 结果展示 -> 训练演示</div>
            <div class="artifact-text">处理后数据目录：{{ showcase.artifacts?.processedDataDir || '-' }}</div>
          </div>
        </div>
      </el-card>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import http from '../api/http'
import MetricCard from '../components/MetricCard.vue'
import SimpleChart from '../components/SimpleChart.vue'

const loading = ref(false)
const errorText = ref('')
const showcase = reactive({
  hasData: false,
  datasetName: 'CIC-IDS-2017',
  flowCount: null,
  pairCount: null,
  topK: null,
  sampleLimit: null,
  bestModel: null,
  models: [],
  conclusion: null,
  artifacts: null
})

const modelNames = computed(() => showcase.models.map((item) => item.displayName))

const prfOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  legend: { data: ['Precision', 'Recall', 'F1'] },
  xAxis: { type: 'category', data: modelNames.value },
  yAxis: { type: 'value', min: 0, max: 1 },
  series: [
    { name: 'Precision', type: 'bar', data: showcase.models.map((item) => Number(item.precision || 0)), itemStyle: { color: '#2563eb' } },
    { name: 'Recall', type: 'bar', data: showcase.models.map((item) => Number(item.recall || 0)), itemStyle: { color: '#0f766e' } },
    { name: 'F1', type: 'bar', data: showcase.models.map((item) => Number(item.f1 || 0)), itemStyle: { color: '#f97316' } }
  ]
}))

const compressionLatencyOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  legend: { data: ['Compression Rate', 'Avg Query Latency(ms)'] },
  xAxis: { type: 'category', data: modelNames.value },
  yAxis: [
    { type: 'value', min: 0, max: 1, name: 'Compression' },
    { type: 'value', min: 0, name: 'Latency(ms)' }
  ],
  series: [
    {
      name: 'Compression Rate',
      type: 'bar',
      data: showcase.models.map((item) => Number(item.compressionRate || 0)),
      itemStyle: { color: '#7c3aed' }
    },
    {
      name: 'Avg Query Latency(ms)',
      type: 'line',
      yAxisIndex: 1,
      smooth: true,
      data: showcase.models.map((item) => Number(item.avgQueryLatencyMs || 0)),
      itemStyle: { color: '#dc2626' }
    }
  ]
}))

function formatRatio(value) {
  if (value === null || value === undefined || value === '') return '-'
  const num = Number(value)
  return `${num.toFixed(4)} (${(num * 100).toFixed(2)}%)`
}

function formatLatency(value) {
  if (value === null || value === undefined || value === '') return '-'
  return `${Number(value).toFixed(2)} ms`
}

function formatLatencyDelta(value) {
  if (value === null || value === undefined || value === '') return '-'
  const num = Number(value)
  return `${num > 0 ? '+' : ''}${num.toFixed(2)} ms`
}

function formatDelta(value) {
  if (value === null || value === undefined || value === '') return '-'
  const num = Number(value)
  return `${num > 0 ? '+' : ''}${num.toFixed(4)} (${num > 0 ? '+' : ''}${(num * 100).toFixed(2)}%)`
}

function formatCount(value) {
  if (value === null || value === undefined || value === '') return '-'
  return Number(value).toLocaleString('zh-CN')
}

async function loadShowcase() {
  loading.value = true
  errorText.value = ''
  try {
    const { data } = await http.get('/experiments/showcase')
    Object.assign(showcase, data)
  } catch (error) {
    errorText.value = error?.response?.data?.message || error?.message || '无法读取成果展示数据'
  } finally {
    loading.value = false
  }
}

onMounted(loadShowcase)
</script>

<style scoped>
.showcase-hero {
  overflow: hidden;
  border: 1px solid rgba(251, 191, 36, 0.3);
  background:
    radial-gradient(circle at top right, rgba(251, 191, 36, 0.18), transparent 28%),
    radial-gradient(circle at left bottom, rgba(37, 99, 235, 0.16), transparent 22%),
    linear-gradient(135deg, #fffdf5 0%, #f8fbff 45%, #f5f9ff 100%);
}

.showcase-hero__content {
  display: grid;
  grid-template-columns: 1.7fr 1fr;
  gap: 18px;
  align-items: stretch;
}

.showcase-kicker {
  font-size: 12px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #b45309;
  font-weight: 700;
}

.showcase-hero h2 {
  margin: 10px 0 12px;
  font-size: 34px;
  line-height: 1.15;
  color: #0f172a;
}

.showcase-hero p {
  margin: 0;
  max-width: 700px;
  color: #475569;
  line-height: 1.8;
}

.showcase-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}

.showcase-hero__callout {
  border-radius: 24px;
  padding: 24px;
  background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
  color: #e2e8f0;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
}

.callout-label {
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #93c5fd;
}

.callout-title {
  margin-top: 14px;
  font-size: 26px;
  font-weight: 700;
  line-height: 1.35;
}

.callout-meta {
  display: grid;
  gap: 8px;
  margin-top: 18px;
  color: #cbd5e1;
}

.insight-stack {
  display: grid;
  gap: 14px;
}

.insight-block {
  border: 1px solid #e2e8f0;
  border-radius: 18px;
  padding: 16px 18px;
  background: #f8fbff;
}

.insight-block.emphasis {
  background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
  border-color: #bfdbfe;
}

.insight-title {
  font-size: 15px;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 8px;
}

.insight-body {
  color: #475569;
  line-height: 1.8;
}

.showcase-table {
  margin-top: 14px;
}

.artifact-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
}

.artifact-panel {
  border-radius: 18px;
  border: 1px solid #dbe7f5;
  background: #f8fbff;
  padding: 16px;
}

.artifact-title {
  margin-bottom: 10px;
  font-size: 15px;
  font-weight: 700;
  color: #0f172a;
}

.artifact-code {
  margin-bottom: 8px;
  padding: 10px 12px;
  border-radius: 12px;
  background: #0f172a;
  color: #e2e8f0;
  font-size: 12px;
  line-height: 1.6;
  word-break: break-all;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
}

.artifact-text {
  margin-bottom: 8px;
  color: #475569;
  line-height: 1.8;
}

@media (max-width: 1100px) {
  .showcase-hero__content,
  .artifact-grid {
    grid-template-columns: 1fr;
  }

  .showcase-hero h2 {
    font-size: 28px;
  }
}
</style>
