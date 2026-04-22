<template>
  <div class="page-stack">
    <el-card shadow="never">
      <template #header>检索任务追踪</template>
      <el-table :data="tasks" size="small">
        <el-table-column prop="id" label="编号" width="70" />
        <el-table-column prop="taskName" label="任务名称" min-width="180" />
        <el-table-column label="模型类型" width="190">
          <template #default="{ row }">{{ modelText(row.modelType) }}</template>
        </el-table-column>
        <el-table-column label="查询方式" width="110">
          <template #default="{ row }">{{ queryModeText(row.queryMode) }}</template>
        </el-table-column>
        <el-table-column label="查询样本" min-width="170">
          <template #default="{ row }">{{ queryValueText(row) }}</template>
        </el-table-column>
        <el-table-column label="范围" width="90">
          <template #default="{ row }">{{ scopeText(row.labelScope) }}</template>
        </el-table-column>
        <el-table-column prop="topK" label="Top-K" width="80" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="compressionRate" label="压缩率" width="100" />
        <el-table-column prop="avgLatencyMs" label="时延(ms)" width="110" />
        <el-table-column prop="createTime" label="创建时间" width="180" />
        <el-table-column label="操作" width="160">
          <template #default="{ row }">
            <el-button type="primary" link @click="runTask(row.id)">启动</el-button>
            <el-button link @click="openDetail(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <div class="metric-grid">
      <MetricCard label="待执行/运行中" :value="pendingCount" hint="便于答辩展示任务状态流转" />
      <MetricCard label="已完成任务" :value="successCount" hint="已完成的近重复检索任务数量" />
      <MetricCard label="最近任务消息" :value="latestMessage" hint="当前后端运行摘要" />
      <MetricCard label="最佳压缩率任务" :value="bestCompressionTask?.taskName || '暂无'" :hint="bestCompressionTask ? `${bestCompressionTask.compressionRate}%` : '等待任务执行'" />
    </div>

    <el-drawer v-model="drawerVisible" title="任务参数快照" size="40%">
      <div v-if="currentTask" class="detail-stack">
        <div><strong>任务名称：</strong>{{ currentTask.taskName }}</div>
        <div><strong>模型类型：</strong>{{ modelText(currentTask.modelType) }}</div>
        <div><strong>查询方式：</strong>{{ queryModeText(currentTask.queryMode) }}</div>
        <div><strong>查询样本：</strong>{{ queryValueText(currentTask) }}</div>
        <div><strong>标签范围：</strong>{{ scopeText(currentTask.labelScope) }}</div>
        <div><strong>Top-K：</strong>{{ currentTask.topK }}</div>
        <div><strong>相似度阈值：</strong>{{ currentTask.similarityThreshold }}</div>
        <div><strong>时间窗口：</strong>{{ currentTask.timeWindow }} 分钟</div>
        <div><strong>保留策略：</strong>{{ currentTask.reservePolicy }}</div>
        <div><strong>哈希长度：</strong>{{ currentTask.hashBits }} 位</div>
        <div><strong>运行摘要：</strong>{{ currentTask.runMessage }}</div>
      </div>
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import http from '../api/http'
import MetricCard from '../components/MetricCard.vue'

const tasks = ref([])
const drawerVisible = ref(false)
const currentTask = ref(null)

const pendingCount = computed(() => tasks.value.filter((item) => item.status !== 'SUCCESS').length)
const successCount = computed(() => tasks.value.filter((item) => item.status === 'SUCCESS').length)
const latestMessage = computed(() => tasks.value[0]?.runMessage || '暂无任务消息')
const bestCompressionTask = computed(() =>
  [...tasks.value]
    .filter((item) => item.compressionRate != null)
    .sort((a, b) => Number(b.compressionRate || 0) - Number(a.compressionRate || 0))[0]
)

function statusType(status) {
  if (status === 'SUCCESS') return 'success'
  if (status === 'RUNNING') return 'warning'
  if (status === 'FAILED') return 'danger'
  return 'info'
}

function modelText(modelType) {
  return modelType === 'mlp' ? 'MLP + DeepLSH（baseline）' : 'Bi-GRU + DeepLSH（论文主模型）'
}

function queryModeText(queryMode) {
  return queryMode === 'sample_id' ? 'sample_id' : 'row_index'
}

function scopeText(scope) {
  return scope === 'all' ? '全量' : '同标签'
}

function queryValueText(task) {
  return task.queryMode === 'sample_id' ? task.sampleId || '-' : task.rowIndex ?? '-'
}

function openDetail(task) {
  currentTask.value = task
  drawerVisible.value = true
}

async function loadTasks() {
  const { data } = await http.get('/tasks')
  tasks.value = data
}

async function runTask(id) {
  try {
    await http.post(`/tasks/${id}/run`)
    ElMessage.success('检索任务已执行')
    await loadTasks()
  } catch (error) {
    await loadTasks()
    ElMessage.error(error?.response?.data?.message || '任务执行失败')
  }
}

onMounted(loadTasks)
</script>
