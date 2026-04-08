<template>
  <div class="page-stack">
    <el-card shadow="never">
      <template #header>去重结果查询</template>
      <div class="toolbar-row">
        <el-select v-model="taskId" class="wide-field" placeholder="按任务筛选" @change="loadResults">
          <el-option label="全部任务" :value="null" />
          <el-option
            v-for="task in tasks"
            :key="task.id"
            :label="`${task.taskName} (#${task.id})`"
            :value="task.id"
          />
        </el-select>
        <el-button @click="loadResults">刷新结果</el-button>
      </div>

      <el-table :data="results" size="small">
        <el-table-column prop="id" label="编号" width="70" />
        <el-table-column prop="logId" label="日志ID" width="90" />
        <el-table-column prop="attackType" label="攻击类型" width="140" />
        <el-table-column prop="similarityScore" label="相似度" width="100" />
        <el-table-column prop="clusterId" label="聚类编号" />
        <el-table-column prop="isRedundant" label="冗余标识" width="100">
          <template #default="{ row }">
            <el-tag :type="row.isRedundant ? 'danger' : 'success'">
              {{ row.isRedundant ? '冗余' : '保留' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button type="primary" link @click="openDetail(row)">详情</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-drawer v-model="drawerVisible" title="聚类详情" size="40%">
      <div v-if="currentResult" class="detail-stack">
        <div><strong>候选样本：</strong>{{ currentResult.candidateSampleId }}</div>
        <div><strong>来源文件：</strong>{{ currentResult.sourceFile }}</div>
        <div><strong>哈希编码：</strong>{{ currentResult.hashCode }}</div>
        <div><strong>代表日志：</strong>{{ currentResult.reserveLogId }}</div>
        <div><strong>聚类编号：</strong>{{ currentResult.clusterId }}</div>
      </div>
    </el-drawer>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import http from '../api/http'

const taskId = ref(null)
const tasks = ref([])
const results = ref([])
const drawerVisible = ref(false)
const currentResult = ref(null)

async function loadTasks() {
  const { data } = await http.get('/tasks')
  tasks.value = data
}

async function loadResults() {
  const { data } = await http.get('/results', { params: { taskId: taskId.value } })
  results.value = data
}

function openDetail(row) {
  currentResult.value = row
  drawerVisible.value = true
}

onMounted(async () => {
  await loadTasks()
  await loadResults()
})
</script>
