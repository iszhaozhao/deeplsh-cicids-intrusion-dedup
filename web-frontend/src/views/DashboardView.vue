<template>
  <div class="page-stack">
    <div class="metric-grid">
      <MetricCard label="任务总数" :value="overview.totalTasks" hint="系统累计去重任务" />
      <MetricCard label="日志总量" :value="overview.totalLogs" hint="任务统计口径汇总" />
      <MetricCard label="平均压缩率" :value="`${overview.avgCompressionRate || 0}%`" hint="去重后冗余压缩表现" />
      <MetricCard label="平均时延" :value="`${overview.avgLatencyMs || 0} ms`" hint="每条日志平均耗时" />
    </div>

    <div class="panel-grid">
      <el-card shadow="never">
        <template #header>最近任务</template>
        <el-table :data="overview.recentTasks" size="small">
          <el-table-column prop="taskName" label="任务名称" />
          <el-table-column prop="status" label="状态" width="120" />
          <el-table-column prop="compressionRate" label="压缩率" width="120" />
          <el-table-column prop="avgLatencyMs" label="时延(ms)" width="120" />
        </el-table>
      </el-card>

      <el-card shadow="never">
        <template #header>攻击类型分布</template>
        <SimpleChart :option="pieOption" />
      </el-card>
    </div>

    <div class="panel-grid single">
      <el-card shadow="never">
        <template #header>平台运行概览</template>
        <SimpleChart :option="barOption" />
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive } from 'vue'
import http from '../api/http'
import MetricCard from '../components/MetricCard.vue'
import SimpleChart from '../components/SimpleChart.vue'

const overview = reactive({
  totalTasks: 0,
  totalLogs: 0,
  avgCompressionRate: 0,
  avgLatencyMs: 0,
  recentTasks: [],
  attackTypes: []
})

const pieOption = computed(() => ({
  tooltip: { trigger: 'item' },
  series: [
    {
      type: 'pie',
      radius: ['45%', '70%'],
      data: overview.attackTypes
    }
  ]
}))

const barOption = computed(() => ({
  tooltip: { trigger: 'axis' },
  xAxis: {
    type: 'category',
    data: overview.recentTasks.map((item) => item.taskName)
  },
  yAxis: { type: 'value' },
  series: [
    {
      name: '压缩率',
      type: 'bar',
      itemStyle: { color: '#4f6bed' },
      data: overview.recentTasks.map((item) => Number(item.compressionRate || 0))
    }
  ]
}))

async function loadOverview() {
  const { data } = await http.get('/stats/overview')
  Object.assign(overview, data)
}

onMounted(loadOverview)
</script>
