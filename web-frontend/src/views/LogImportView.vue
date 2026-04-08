<template>
  <div class="page-stack">
    <el-card shadow="never">
      <template #header>导入日志文件</template>
      <el-form :model="form" label-width="110px">
        <el-form-item label="关联任务">
          <el-select v-model="form.taskId" class="wide-field" placeholder="请选择任务">
            <el-option
              v-for="task in tasks"
              :key="task.id"
              :label="`${task.taskName} (#${task.id})`"
              :value="task.id"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="CSV 文件">
          <input type="file" accept=".csv" @change="onFileChange" />
        </el-form-item>
        <el-button type="primary" :disabled="!file || !form.taskId" @click="upload">上传并预览</el-button>
      </el-form>
    </el-card>

    <el-card shadow="never" v-if="uploadResult">
      <template #header>导入反馈</template>
      <div class="upload-summary">
        <div>文件名：{{ uploadResult.fileName }}</div>
        <div>任务编号：{{ uploadResult.taskId }}</div>
        <div>总行数：{{ uploadResult.totalRows }}</div>
        <div>结果：{{ uploadResult.message }}</div>
      </div>
      <el-table :data="uploadResult.previewRows" size="small">
        <el-table-column
          v-for="(header, index) in uploadResult.headers"
          :key="header"
          :prop="String(index)"
          :label="header"
        />
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import http from '../api/http'

const tasks = ref([])
const file = ref(null)
const uploadResult = ref(null)
const form = reactive({
  taskId: null
})

function onFileChange(event) {
  file.value = event.target.files[0]
}

async function loadTasks() {
  const { data } = await http.get('/tasks')
  tasks.value = data
  if (!form.taskId && data.length) {
    form.taskId = data[0].id
  }
}

async function upload() {
  if (!file.value || !form.taskId) return
  const formData = new FormData()
  formData.append('taskId', form.taskId)
  formData.append('file', file.value)
  try {
    const { data } = await http.post('/logs/upload', formData)
    uploadResult.value = data
    ElMessage.success('日志上传成功')
  } catch (error) {
    ElMessage.error(error?.response?.data?.message || '上传失败')
  }
}

onMounted(loadTasks)
</script>
