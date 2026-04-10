<template>
  <div class="page-stack">
    <div class="metric-grid">
      <MetricCard label="检索任务数" :value="overview.recentQueryTasks || overview.totalTasks" hint="已创建并可用于近重复检索的任务" />
      <MetricCard label="平均压缩率" :value="`${overview.avgCompressionRate || 0}%`" hint="日志去重后的整体压缩表现" />
      <MetricCard label="平均查询时延" :value="`${overview.avgLatencyMs || 0} ms`" hint="每条候选日志的平均检索耗时" />
      <MetricCard
        label="当前最佳模型"
        :value="experimentSummary.bestModelDisplayName || '暂无离线评估结果'"
        :hint="experimentSummary.bestModelF1 ? `F1 = ${formatNumber(experimentSummary.bestModelF1)}` : '请准备离线评估结果文件'"
      />
    </div>

    <div class="panel-grid">
      <el-card shadow="never" class="hero-panel">
        <template #header>系统定位</template>
        <div class="detail-stack">
          <div>本系统用于展示 <strong>网络入侵日志去重、近重复检索与告警聚合</strong>，不是普通入侵分类平台。</div>
          <div>当前推荐演示主线：<strong>检索任务配置 → 任务执行 → 近重复结果 → 实验对比分析</strong></div>
          <div>最近任务模型：{{ modelText(overview.latestModelType) || '暂无任务' }}</div>
          <div>最近任务检索范围：{{ scopeText(overview.latestLabelScope) || '-' }}</div>
          <div>最近任务 Top-K：{{ overview.latestTopK ?? '-' }}</div>
        </div>
      </el-card>

      <el-card shadow="never">
        <template #header>论文模型摘要</template>
        <div class="detail-stack">
          <div><strong>Baseline：</strong>MLP + DeepLSH</div>
          <div><strong>主模型：</strong>Bi-GRU + DeepLSH</div>
          <div><strong>评估来源：</strong>离线评估结果文件</div>
          <div><strong>最佳模型：</strong>{{ experimentSummary.bestModelDisplayName || '暂无' }}</div>
          <div><strong>最佳 F1：</strong>{{ experimentSummary.bestModelF1 ? formatNumber(experimentSummary.bestModelF1) : '-' }}</div>
        </div>
      </el-card>
    </div>

    <div class="panel-grid">
      <el-card shadow="never">
        <template #header>最近任务压缩率与时延</template>
        <SimpleChart :option="performanceOption" />
      </el-card>
      <el-card shadow="never">
        <template #header>标签分布参考</template>
        <SimpleChart :option="labelOption" />
      </el-card>
    </div>

    <div class="panel-grid single">
      <el-card shadow="never">
        <template #header>最近检索任务概览</template>
        <el-table :data="overview.recentTasks" size="small">
          <el-table-column prop="taskName" label="任务名称" min-width="180" />
          <el-table-column label="模型类型" width="190">
            <template #default="{ row }">{{ modelText(row.modelType) }}</template>
          </el-table-column>
          <el-table-column label="查询方式" width="120">
            <template #default="{ row }">{{ queryModeText(row.queryMode) }}</template>
          </el-table-column>
          <el-table-column label="范围" width="100">
            <template #default="{ row }">{{ scopeText(row.labelScope) }}</template>
          </el-table-column>
          <el-table-column prop="topK" label="Top-K" width="90" />
          <el-table-column prop="status" label="状态" width="100" />
          <el-table-column prop="compressionRate" label="压缩率" width="100" />
          <el-table-column prop="avgLatencyMs" label="时延(ms)" width="110" />
        </el-table>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import http from '../api/http'
import MetricCard from '../components/MetricCard.vue'
import SimpleChart from '../components/SimpleChart.vue'

const overview = reactive({
  totalTasks: 0,
  totalLogs: 0,
  totalResults: 0,
  recentQueryTasks: 0,
  avgCompressionRate: 0,
  avgLatencyMs: 0,
  latestModelType: null,
  latestLabelScope: null,
  latestTopK: null,
  bestModelName: null,
  bestModelDisplayName: null,
  bestModelF1: null,
  recentTasks: [],
  attackTypes: []
})

const experimentSummary = reactive({
  bestModelDisplayName: null,
  bestModelF1: null
})

const labelStats = ref([])

const performanceOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  legend: { data: ['压缩率', '查询时延'] },
  xAxis: { type: 'category', data: overview.recentTasks.map((item) => item.taskName) },
  yAxis: [
    { type: 'value', name: '压缩率' },
    { type: 'value', name: '时延(ms)' }
  ],
  series: [
    {
      name: '压缩率',
      type: 'bar',
      data: overview.recentTasks.map((item) => Number(item.compressionRate || 0)),
      itemStyle: { color: '#3b82f6' }
    },
    {
      name: '查询时延',
      type: 'line',
      yAxisIndex: 1,
      smooth: true,
      data: overview.recentTasks.map((item) => Number(item.avgLatencyMs || 0)),
      itemStyle: { color: '#0f766e' }
    }
  ]
}))

const labelOption = computed(() => ({
  tooltip: { trigger: 'item' },
  series: [
    {
      type: 'pie',
      radius: ['40%', '68%'],
      data: labelStats.value
    }
  ]
}))

function modelText(modelType) {
  if (modelType === 'mlp') return 'MLP + DeepLSH（baseline）'
  if (modelType === 'bigru') return 'Bi-GRU + DeepLSH（论文主模型）'
  return modelType || '-'
}

function queryModeText(queryMode) {
  return queryMode === 'sample_id' ? 'sample_id' : 'row_index'
}

function scopeText(scope) {
  if (scope === 'same') return '同标签'
  if (scope === 'all') return '全量'
  return scope || '-'
}

function formatNumber(value) {
  return Number(value).toFixed(4)
}

async function loadOverview() {
  const { data } = await http.get('/stats/overview')
  Object.assign(overview, data)
}

async function loadSummary() {
  const { data } = await http.get('/experiments/summary')
  Object.assign(experimentSummary, data)
}

async function loadLabels() {
  const { data } = await http.get('/meta/labels')
  labelStats.value = data
}

onMounted(async () => {
  await Promise.all([loadOverview(), loadSummary(), loadLabels()])
})
</script>
