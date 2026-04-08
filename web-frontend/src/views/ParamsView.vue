<template>
  <div class="page-stack">
    <el-card shadow="never">
      <template #header>创建去重参数任务</template>
      <el-form :model="form" label-width="140px">
        <el-form-item label="任务名称">
          <el-input v-model="form.taskName" class="wide-field" />
        </el-form-item>
        <el-form-item label="相似度阈值">
          <el-input-number v-model="form.similarityThreshold" :min="0.1" :max="1" :step="0.01" />
        </el-form-item>
        <el-form-item label="时间窗口(分钟)">
          <el-input-number v-model="form.timeWindow" :min="1" :max="1440" />
        </el-form-item>
        <el-form-item label="保留策略">
          <el-select v-model="form.reservePolicy" class="wide-field">
            <el-option label="保留最早日志" value="EARLIEST" />
            <el-option label="保留最新日志" value="LATEST" />
            <el-option label="保留高风险日志" value="HIGHEST_RISK" />
          </el-select>
        </el-form-item>
        <el-form-item label="哈希编码长度">
          <el-select v-model="form.hashBits" class="wide-field">
            <el-option label="32 位" :value="32" />
            <el-option label="64 位" :value="64" />
          </el-select>
        </el-form-item>
        <el-form-item label="查询样本 rowIndex">
          <el-input-number v-model="form.rowIndex" :min="0" :max="200" />
        </el-form-item>
        <el-form-item label="标签范围">
          <el-select v-model="form.labelScope" class="wide-field">
            <el-option label="同标签" value="same" />
            <el-option label="全量候选" value="all" />
          </el-select>
        </el-form-item>
        <el-form-item label="返回候选数">
          <el-input-number v-model="form.topK" :min="1" :max="20" />
        </el-form-item>
        <el-button type="primary" @click="createTask">保存任务参数</el-button>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { reactive } from 'vue'
import { ElMessage } from 'element-plus'
import http from '../api/http'

const form = reactive({
  taskName: 'Web 原型演示任务',
  similarityThreshold: 0.85,
  timeWindow: 60,
  reservePolicy: 'EARLIEST',
  hashBits: 32,
  rowIndex: 0,
  labelScope: 'same',
  topK: 10
})

async function createTask() {
  try {
    await http.post('/tasks', form)
    ElMessage.success('参数任务已创建')
  } catch (error) {
    ElMessage.error(error?.response?.data?.message || '任务创建失败')
  }
}
</script>
