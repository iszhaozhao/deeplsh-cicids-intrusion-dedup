<template>
  <div class="page-stack">
    <el-card shadow="never">
      <template #header>创建可复现的近重复检索任务</template>
      <el-form :model="form" label-width="130px">
        <div class="form-grid">
          <el-form-item label="任务名称">
            <el-input v-model="form.taskName" class="wide-field" />
          </el-form-item>
          <el-form-item label="模型类型">
            <el-segmented v-model="form.modelType" :options="modelOptions" class="wide-field" />
          </el-form-item>
          <el-form-item label="查询方式">
            <el-segmented v-model="form.queryMode" :options="queryModeOptions" class="wide-field" />
          </el-form-item>
          <el-form-item label="标签范围">
            <el-segmented v-model="form.labelScope" :options="labelScopeOptions" class="wide-field" />
          </el-form-item>
          <el-form-item v-if="form.queryMode === 'sample_id'" label="查询样本 ID">
            <el-input v-model="form.sampleId" class="wide-field" placeholder="例如 Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv#0" />
          </el-form-item>
          <el-form-item v-else label="查询样本 rowIndex">
            <el-input-number v-model="form.rowIndex" :min="0" :max="20000" class="wide-field" />
          </el-form-item>
          <el-form-item label="返回候选数 Top-K">
            <el-input-number v-model="form.topK" :min="1" :max="20" />
          </el-form-item>
          <el-form-item label="相似度阈值">
            <el-input-number v-model="form.similarityThreshold" :min="0.1" :max="1" :step="0.01" />
          </el-form-item>
        </div>

        <el-collapse>
          <el-collapse-item title="高级参数（答辩时可选展开）" name="advanced">
            <div class="form-grid">
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
            </div>
          </el-collapse-item>
        </el-collapse>

        <div class="toolbar-row">
          <el-button type="primary" @click="createTask">保存检索任务</el-button>
          <el-button @click="resetForm">恢复默认参数</el-button>
        </div>
      </el-form>
    </el-card>

    <el-card shadow="never">
      <template #header>当前配置说明</template>
      <div class="detail-stack">
        <div>当前将使用 <strong>{{ modelText(form.modelType) }}</strong> 执行近重复检索。</div>
        <div>查询入口为 <strong>{{ queryModeText(form.queryMode) }}</strong>，查询值为 <strong>{{ queryValueText }}</strong>。</div>
        <div>检索范围设置为 <strong>{{ scopeText(form.labelScope) }}</strong>，系统返回 <strong>Top {{ form.topK }}</strong> 候选样本。</div>
        <div>相似度阈值为 <strong>{{ form.similarityThreshold }}</strong>，高于阈值的候选将被判为冗余。</div>
        <div>高级参数采用 <strong>{{ form.reservePolicy }}</strong> 保留策略，哈希长度为 <strong>{{ form.hashBits }} 位</strong>。</div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { computed, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import http from '../api/http'

const modelOptions = [
  { label: 'Bi-GRU 主模型', value: 'bigru' },
  { label: 'MLP Baseline', value: 'mlp' }
]

const queryModeOptions = [
  { label: 'row_index', value: 'row_index' },
  { label: 'sample_id', value: 'sample_id' }
]

const labelScopeOptions = [
  { label: '同标签', value: 'same' },
  { label: '全量', value: 'all' }
]

const form = reactive(defaultForm())

const queryValueText = computed(() => {
  if (form.queryMode === 'sample_id') {
    return form.sampleId || '未填写 sample_id'
  }
  return String(form.rowIndex ?? 0)
})

function defaultForm() {
  return {
    taskName: 'Bi-GRU 检索演示任务',
    modelType: 'bigru',
    queryMode: 'row_index',
    sampleId: '',
    rowIndex: 0,
    labelScope: 'same',
    topK: 10,
    similarityThreshold: 0.85,
    timeWindow: 60,
    reservePolicy: 'EARLIEST',
    hashBits: 32
  }
}

function modelText(modelType) {
  return modelType === 'mlp' ? 'MLP + DeepLSH（baseline）' : 'Bi-GRU + DeepLSH（论文主模型）'
}

function queryModeText(queryMode) {
  return queryMode === 'sample_id' ? 'sample_id' : 'row_index'
}

function scopeText(scope) {
  return scope === 'all' ? '全量候选' : '同标签候选'
}

function resetForm() {
  Object.assign(form, defaultForm())
}

async function createTask() {
  const payload = {
    taskName: form.taskName,
    modelType: form.modelType,
    similarityThreshold: form.similarityThreshold,
    timeWindow: form.timeWindow,
    reservePolicy: form.reservePolicy,
    hashBits: form.hashBits,
    sampleId: form.queryMode === 'sample_id' ? form.sampleId : null,
    rowIndex: form.queryMode === 'row_index' ? form.rowIndex : null,
    labelScope: form.labelScope,
    topK: form.topK
  }

  if (form.queryMode === 'sample_id' && !form.sampleId) {
    ElMessage.warning('请输入 sample_id 作为查询样本')
    return
  }

  try {
    await http.post('/tasks', payload)
    ElMessage.success('检索任务已创建')
  } catch (error) {
    ElMessage.error(error?.response?.data?.message || '任务创建失败')
  }
}
</script>
