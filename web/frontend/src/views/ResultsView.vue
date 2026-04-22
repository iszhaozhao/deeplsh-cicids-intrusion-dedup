<template>
  <div class="page-stack">
    <el-card shadow="never">
      <template #header>近重复检索结果与告警聚合</template>
      <div class="toolbar-row">
        <el-select v-model="taskId" class="wide-field" placeholder="按任务筛选" @change="handleTaskChange">
          <el-option
            v-for="task in tasks"
            :key="task.id"
            :label="`${task.taskName} (#${task.id})`"
            :value="task.id"
          />
        </el-select>
        <el-select v-model="compareTaskId" class="wide-field" clearable placeholder="选择对比任务（可选）" @change="loadCompareResults">
          <el-option
            v-for="task in compareTaskOptions"
            :key="task.id"
            :label="`${task.taskName} (#${task.id})`"
            :value="task.id"
          />
        </el-select>
        <el-button @click="loadResults">刷新结果</el-button>
      </div>

      <div v-if="selectedTask" class="result-summary">
        <el-tag>{{ modelText(selectedTask.modelType) }}</el-tag>
        <el-tag type="info">{{ queryModeText(selectedTask.queryMode) }}</el-tag>
        <el-tag type="success">{{ scopeText(selectedTask.labelScope) }}</el-tag>
        <el-tag type="warning">Top {{ selectedTask.topK }}</el-tag>
        <span>查询样本：{{ queryValueText(selectedTask) }}</span>
      </div>

      <div v-if="selectedTask && selectedCompareTask" class="result-summary result-summary--compare">
        <el-tag type="danger">{{ modelText(selectedCompareTask.modelType) }}</el-tag>
        <el-tag type="info">{{ queryModeText(selectedCompareTask.queryMode) }}</el-tag>
        <el-tag type="success">{{ scopeText(selectedCompareTask.labelScope) }}</el-tag>
        <el-tag type="warning">Top {{ selectedCompareTask.topK }}</el-tag>
        <span>对比任务查询样本：{{ queryValueText(selectedCompareTask) }}</span>
      </div>

      <div v-if="selectedCompareTask" class="metric-grid">
        <div class="metric-card">
          <div class="metric-label">Top1 是否一致</div>
          <div class="metric-value metric-value--compact">{{ comparisonSummary.sameTop1 ? '相同' : '不同' }}</div>
          <div class="metric-hint">判断两个模型的首个候选是否一致</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Top-K 重合数</div>
          <div class="metric-value metric-value--compact">{{ comparisonSummary.overlapCount }}/{{ Math.max(results.length, compareResults.length) }}</div>
          <div class="metric-hint">两个任务返回候选样本的交集规模</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">当前任务特有候选</div>
          <div class="metric-value metric-value--compact">{{ comparisonSummary.onlyPrimaryCount }}</div>
          <div class="metric-hint">只在当前任务里出现的候选样本</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">对比任务特有候选</div>
          <div class="metric-value metric-value--compact">{{ comparisonSummary.onlyCompareCount }}</div>
          <div class="metric-hint">只在对比任务里出现的候选样本</div>
        </div>
      </div>

      <el-table :data="results" size="small">
        <el-table-column prop="querySampleId" label="query_sample_id" min-width="180" />
        <el-table-column prop="candidateSampleId" label="candidate_sample_id" min-width="180" />
        <el-table-column prop="queryLabel" label="query_label" width="130" />
        <el-table-column prop="candidateLabel" label="candidate_label" width="140" />
        <el-table-column prop="similarityScore" label="embedding_similarity" width="150" />
        <el-table-column prop="hashBucketHits" label="hash_bucket_hits" width="140" />
        <el-table-column prop="isSameLabel" label="is_same_label" width="120">
          <template #default="{ row }">{{ row.isSameLabel ? '是' : '否' }}</template>
        </el-table-column>
        <el-table-column prop="isRedundant" label="is_redundant" width="120">
          <template #default="{ row }">
            <el-tag :type="row.isRedundant ? 'danger' : 'success'">{{ row.isRedundant ? '冗余' : '保留' }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="sourceFile" label="source_file" min-width="180" />
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button type="primary" link @click="openDetail(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <div v-if="selectedCompareTask" class="panel-grid">
      <el-card shadow="never">
        <template #header>当前任务独有候选</template>
        <el-table :data="onlyPrimaryRows" size="small">
          <el-table-column prop="candidateSampleId" label="candidate_sample_id" min-width="180" />
          <el-table-column prop="candidateLabel" label="candidate_label" width="140" />
          <el-table-column prop="similarityScore" label="embedding_similarity" width="150" />
          <el-table-column prop="sourceFile" label="source_file" min-width="180" />
        </el-table>
      </el-card>
      <el-card shadow="never">
        <template #header>对比任务独有候选</template>
        <el-table :data="onlyCompareRows" size="small">
          <el-table-column prop="candidateSampleId" label="candidate_sample_id" min-width="180" />
          <el-table-column prop="candidateLabel" label="candidate_label" width="140" />
          <el-table-column prop="similarityScore" label="embedding_similarity" width="150" />
          <el-table-column prop="sourceFile" label="source_file" min-width="180" />
        </el-table>
      </el-card>
    </div>

    <div class="panel-grid">
      <el-card shadow="never">
        <template #header>Top-K 相似度分布</template>
        <SimpleChart :option="similarityOption" />
      </el-card>
      <el-card shadow="never">
        <template #header>哈希桶命中分布</template>
        <SimpleChart :option="bucketOption" />
      </el-card>
    </div>

    <div class="panel-grid">
      <el-card shadow="never">
        <template #header>冗余 / 保留占比</template>
        <SimpleChart :option="redundantOption" />
      </el-card>
      <el-card shadow="never">
        <template #header>候选标签分布</template>
        <SimpleChart :option="candidateLabelOption" />
      </el-card>
    </div>

    <el-drawer v-model="drawerVisible" title="检索结果详情" size="42%">
      <div v-if="currentResult" class="detail-stack">
        <div><strong>查询样本：</strong>{{ currentResult.querySampleId }}</div>
        <div><strong>候选样本：</strong>{{ currentResult.candidateSampleId }}</div>
        <div><strong>查询标签：</strong>{{ currentResult.queryLabel }}</div>
        <div><strong>候选标签：</strong>{{ currentResult.candidateLabel }}</div>
        <div><strong>相似度：</strong>{{ currentResult.similarityScore }}</div>
        <div><strong>哈希桶命中数：</strong>{{ currentResult.hashBucketHits }}</div>
        <div><strong>是否同标签：</strong>{{ currentResult.isSameLabel ? '是' : '否' }}</div>
        <div><strong>冗余判定：</strong>{{ currentResult.isRedundant ? '冗余' : '保留' }}</div>
        <div><strong>代表日志：</strong>{{ currentResult.reserveLogId }}</div>
        <div><strong>聚类编号：</strong>{{ currentResult.clusterId }}</div>
        <div><strong>哈希编码：</strong>{{ currentResult.hashCode }}</div>
        <div><strong>来源文件：</strong>{{ currentResult.sourceFile }}</div>
      </div>
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import http from '../api/http'
import SimpleChart from '../components/SimpleChart.vue'

const taskId = ref(null)
const compareTaskId = ref(null)
const tasks = ref([])
const results = ref([])
const compareResults = ref([])
const drawerVisible = ref(false)
const currentResult = ref(null)

const selectedTask = computed(() => tasks.value.find((item) => item.id === taskId.value) || null)
const selectedCompareTask = computed(() => tasks.value.find((item) => item.id === compareTaskId.value) || null)
const compareTaskOptions = computed(() => tasks.value.filter((item) => item.id !== taskId.value))

const comparisonSummary = computed(() => {
  const primaryIds = results.value.map((item) => item.candidateSampleId)
  const compareIds = compareResults.value.map((item) => item.candidateSampleId)
  const overlap = primaryIds.filter((item) => compareIds.includes(item))
  return {
    sameTop1: primaryIds[0] && compareIds[0] && primaryIds[0] === compareIds[0],
    overlapCount: overlap.length,
    onlyPrimaryCount: primaryIds.filter((item) => !compareIds.includes(item)).length,
    onlyCompareCount: compareIds.filter((item) => !primaryIds.includes(item)).length
  }
})

const onlyPrimaryRows = computed(() => {
  const compareIds = new Set(compareResults.value.map((item) => item.candidateSampleId))
  return results.value.filter((item) => !compareIds.has(item.candidateSampleId))
})

const onlyCompareRows = computed(() => {
  const primaryIds = new Set(results.value.map((item) => item.candidateSampleId))
  return compareResults.value.filter((item) => !primaryIds.has(item.candidateSampleId))
})

const similarityOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  xAxis: { type: 'category', data: results.value.map((item) => item.candidateSampleId) },
  yAxis: { type: 'value', min: 0 },
  series: [{ type: 'bar', data: results.value.map((item) => Number(item.similarityScore || 0)), itemStyle: { color: '#2563eb' } }]
}))

const bucketOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  xAxis: { type: 'category', data: results.value.map((item) => item.candidateSampleId) },
  yAxis: { type: 'value', min: 0 },
  series: [{ type: 'bar', data: results.value.map((item) => Number(item.hashBucketHits || 0)), itemStyle: { color: '#0f766e' } }]
}))

const redundantOption = computed(() => {
  const redundant = results.value.filter((item) => item.isRedundant).length
  const reserved = results.value.length - redundant
  return {
    tooltip: { trigger: 'item' },
    series: [
      {
        type: 'pie',
        radius: '65%',
        data: [
          { name: '冗余', value: redundant },
          { name: '保留', value: reserved }
        ]
      }
    ]
  }
})

const candidateLabelOption = computed(() => {
  const counts = results.value.reduce((acc, item) => {
    const key = item.candidateLabel || item.attackType || 'UNKNOWN'
    acc[key] = (acc[key] || 0) + 1
    return acc
  }, {})
  return {
    tooltip: { trigger: 'item' },
    series: [
      {
        type: 'pie',
        radius: ['35%', '68%'],
        data: Object.entries(counts).map(([name, value]) => ({ name, value }))
      }
    ]
  }
})

function modelText(modelType) {
  return modelType === 'mlp' ? 'MLP + DeepLSH（baseline）' : 'Bi-GRU + DeepLSH（论文主模型）'
}

function queryModeText(queryMode) {
  return queryMode === 'sample_id' ? 'sample_id' : 'row_index'
}

function scopeText(scope) {
  return scope === 'all' ? '全量候选' : '同标签候选'
}

function queryValueText(task) {
  return task.queryMode === 'sample_id' ? task.sampleId || '-' : task.rowIndex ?? '-'
}

function openDetail(row) {
  currentResult.value = row
  drawerVisible.value = true
}

async function loadTasks() {
  const { data } = await http.get('/tasks')
  tasks.value = data
  if (!taskId.value && data.length) {
    taskId.value = data[0].id
  }
}

async function loadResults() {
  if (!taskId.value) {
    results.value = []
    return
  }
  const { data } = await http.get('/results', { params: { taskId: taskId.value } })
  results.value = data
}

async function loadCompareResults() {
  if (!compareTaskId.value) {
    compareResults.value = []
    return
  }
  const { data } = await http.get('/results', { params: { taskId: compareTaskId.value } })
  compareResults.value = data
}

async function handleTaskChange() {
  if (compareTaskId.value === taskId.value) {
    compareTaskId.value = null
    compareResults.value = []
  }
  await loadResults()
}

onMounted(async () => {
  await loadTasks()
  await loadResults()
})
</script>
