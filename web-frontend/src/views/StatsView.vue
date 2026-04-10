<template>
  <div class="page-stack">
    <div class="metric-grid">
      <MetricCard label="最佳模型" :value="summary.bestModelDisplayName || '暂无离线结果'" hint="论文实验对比中的当前最佳模型" />
      <MetricCard label="最佳 F1" :value="summary.bestModelF1 ?? '-'" hint="由离线评估脚本汇总得出" />
      <MetricCard label="评估 Top-K" :value="summary.topK ?? '-'" hint="离线评估采用的候选返回数" />
      <MetricCard label="评估样本数" :value="summary.sampleLimit ?? '-'" hint="离线实验采样规模" />
    </div>

    <el-card shadow="never" v-if="!metrics.length">
      <el-empty description="未检测到离线评估结果，请先准备 results 目录中的指标文件。" />
    </el-card>

    <template v-else>
      <div class="panel-grid">
        <el-card shadow="never">
          <template #header>Precision / Recall / F1 对比</template>
          <SimpleChart :option="prfOption" />
        </el-card>
        <el-card shadow="never">
          <template #header>压缩率对比</template>
          <SimpleChart :option="compressionOption" />
        </el-card>
      </div>

      <div class="panel-grid">
        <el-card shadow="never">
          <template #header>查询时延对比</template>
          <SimpleChart :option="latencyOption" />
        </el-card>
        <el-card shadow="never">
          <template #header>最佳模型摘要</template>
          <div class="detail-stack">
            <div><strong>最佳模型：</strong>{{ summary.bestModelDisplayName }}</div>
            <div><strong>F1：</strong>{{ summary.bestModelF1 }}</div>
            <div><strong>Precision：</strong>{{ summary.bestModelPrecision }}</div>
            <div><strong>Recall：</strong>{{ summary.bestModelRecall }}</div>
            <div><strong>压缩率：</strong>{{ summary.bestModelCompressionRate }}</div>
            <div><strong>查询时延：</strong>{{ summary.bestModelAvgQueryLatencyMs }} ms</div>
          </div>
        </el-card>
      </div>

      <el-card shadow="never">
        <template #header>实验指标明细</template>
        <el-table :data="metrics" size="small">
          <el-table-column prop="displayName" label="模型" min-width="220" />
          <el-table-column prop="accuracy" label="Accuracy" width="110" />
          <el-table-column prop="precision" label="Precision" width="110" />
          <el-table-column prop="recall" label="Recall" width="110" />
          <el-table-column prop="f1" label="F1" width="100" />
          <el-table-column prop="compressionRate" label="Compression Rate" width="150" />
          <el-table-column prop="avgQueryLatencyMs" label="Avg Query Latency(ms)" width="180" />
          <el-table-column prop="threshold" label="Threshold" width="110" />
        </el-table>
      </el-card>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import http from '../api/http'
import MetricCard from '../components/MetricCard.vue'
import SimpleChart from '../components/SimpleChart.vue'

const summary = reactive({
  bestModelDisplayName: null,
  bestModelF1: null,
  bestModelPrecision: null,
  bestModelRecall: null,
  bestModelCompressionRate: null,
  bestModelAvgQueryLatencyMs: null,
  topK: null,
  sampleLimit: null
})

const metrics = ref([])

const modelNames = computed(() => metrics.value.map((item) => item.displayName))

const prfOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  legend: { data: ['Precision', 'Recall', 'F1'] },
  xAxis: { type: 'category', data: modelNames.value },
  yAxis: { type: 'value', min: 0, max: 1 },
  series: [
    { name: 'Precision', type: 'bar', data: metrics.value.map((item) => Number(item.precision || 0)), itemStyle: { color: '#2563eb' } },
    { name: 'Recall', type: 'bar', data: metrics.value.map((item) => Number(item.recall || 0)), itemStyle: { color: '#0f766e' } },
    { name: 'F1', type: 'bar', data: metrics.value.map((item) => Number(item.f1 || 0)), itemStyle: { color: '#f59e0b' } }
  ]
}))

const compressionOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  xAxis: { type: 'category', data: modelNames.value },
  yAxis: { type: 'value' },
  series: [{ type: 'bar', data: metrics.value.map((item) => Number(item.compressionRate || 0)), itemStyle: { color: '#6366f1' } }]
}))

const latencyOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  xAxis: { type: 'category', data: modelNames.value },
  yAxis: { type: 'value' },
  series: [{ type: 'line', smooth: true, data: metrics.value.map((item) => Number(item.avgQueryLatencyMs || 0)), itemStyle: { color: '#ef4444' } }]
}))

async function loadStats() {
  const [summaryResponse, metricsResponse] = await Promise.all([
    http.get('/experiments/summary'),
    http.get('/experiments/metrics')
  ])
  Object.assign(summary, summaryResponse.data)
  metrics.value = metricsResponse.data
}

onMounted(loadStats)
</script>
