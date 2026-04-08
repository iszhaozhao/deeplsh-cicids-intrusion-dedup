<template>
  <div class="page-stack">
    <div class="panel-grid">
      <el-card shadow="never">
        <template #header>压缩率柱状图</template>
        <SimpleChart :option="compressionOption" />
      </el-card>
      <el-card shadow="never">
        <template #header>攻击类型饼图</template>
        <SimpleChart :option="pieOption" />
      </el-card>
    </div>

    <div class="panel-grid">
      <el-card shadow="never">
        <template #header>任务时延折线图</template>
        <SimpleChart :option="latencyOption" />
      </el-card>
      <el-card shadow="never">
        <template #header>冗余趋势图</template>
        <SimpleChart :option="trendOption" />
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive } from 'vue'
import http from '../api/http'
import SimpleChart from '../components/SimpleChart.vue'

const overview = reactive({
  recentTasks: [],
  attackTypes: []
})

const taskNames = computed(() => overview.recentTasks.map((item) => item.taskName))

const compressionOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  xAxis: { type: 'category', data: taskNames.value },
  yAxis: { type: 'value' },
  series: [{ type: 'bar', data: overview.recentTasks.map((item) => Number(item.compressionRate || 0)), itemStyle: { color: '#3b82f6' } }]
}))

const pieOption = computed(() => ({
  tooltip: { trigger: 'item' },
  series: [{ type: 'pie', radius: '65%', data: overview.attackTypes }]
}))

const latencyOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  xAxis: { type: 'category', data: taskNames.value },
  yAxis: { type: 'value' },
  series: [{ type: 'line', smooth: true, data: overview.recentTasks.map((item) => Number(item.avgLatencyMs || 0)), itemStyle: { color: '#0f766e' } }]
}))

const trendOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  xAxis: { type: 'category', data: taskNames.value },
  yAxis: { type: 'value' },
  series: [{ type: 'line', smooth: true, areaStyle: {}, data: overview.recentTasks.map((item) => Number(item.compressionRate || 0)), itemStyle: { color: '#6366f1' } }]
}))

async function loadStats() {
  const { data } = await http.get('/stats/overview')
  Object.assign(overview, data)
}

onMounted(loadStats)
</script>
