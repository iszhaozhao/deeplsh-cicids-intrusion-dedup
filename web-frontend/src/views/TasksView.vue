<template>
  <div class="page-stack">
    <el-card shadow="never">
      <template #header>任务列表与执行状态</template>
      <el-table :data="tasks" size="small">
        <el-table-column prop="id" label="编号" width="80" />
        <el-table-column prop="taskName" label="任务名称" />
        <el-table-column prop="status" label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="similarityThreshold" label="阈值" width="100" />
        <el-table-column prop="compressionRate" label="压缩率" width="120" />
        <el-table-column prop="avgLatencyMs" label="时延(ms)" width="120" />
        <el-table-column label="操作" width="140">
          <template #default="{ row }">
            <el-button type="primary" link @click="runTask(row.id)">启动</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <div class="metric-grid">
      <MetricCard label="待执行/运行中" :value="pendingCount" hint="便于演示任务状态流转" />
      <MetricCard label="已完成任务" :value="successCount" hint="成功执行的去重任务" />
      <MetricCard label="最近一次消息" :value="latestMessage" hint="后端运行摘要" />
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import http from '../api/http'
import MetricCard from '../components/MetricCard.vue'

const tasks = ref([])

const pendingCount = computed(() => tasks.value.filter((item) => item.status !== 'SUCCESS').length)
const successCount = computed(() => tasks.value.filter((item) => item.status === 'SUCCESS').length)
const latestMessage = computed(() => tasks.value[0]?.runMessage || '暂无任务消息')

function statusType(status) {
  if (status === 'SUCCESS') return 'success'
  if (status === 'RUNNING') return 'warning'
  if (status === 'FAILED') return 'danger'
  return 'info'
}

async function loadTasks() {
  const { data } = await http.get('/tasks')
  tasks.value = data
}

async function runTask(id) {
  try {
    await http.post(`/tasks/${id}/run`)
    ElMessage.success('任务已执行')
    await loadTasks()
  } catch (error) {
    ElMessage.error(error?.response?.data?.message || '任务执行失败')
  }
}

onMounted(loadTasks)
</script>
