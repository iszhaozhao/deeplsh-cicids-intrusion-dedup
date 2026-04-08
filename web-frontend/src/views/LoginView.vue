<template>
  <div class="login-page">
    <div class="login-panel">
      <div class="login-header">
        <div class="login-mark">DH</div>
        <div>
          <h1>网络入侵检测日志去重系统</h1>
          <p>Spring Boot + Vue + Python 深度哈希查询原型</p>
        </div>
      </div>

      <el-form :model="form" label-position="top" @submit.prevent="submit">
        <el-form-item label="用户名">
          <el-input v-model="form.username" placeholder="请输入用户名" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="form.password" type="password" placeholder="请输入密码" show-password />
        </el-form-item>
        <el-form-item label="角色">
          <el-select v-model="form.role" class="full-width">
            <el-option label="管理员" value="ADMIN" />
            <el-option label="运维人员" value="OPERATOR" />
          </el-select>
        </el-form-item>
        <el-button type="primary" class="full-width" :loading="loading" @click="submit">登录系统</el-button>
      </el-form>

      <div class="login-tip">
        <div>管理员：`admin / admin123 / ADMIN`</div>
        <div>运维人员：`ops / ops123 / OPERATOR`</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()
const router = useRouter()
const loading = ref(false)

const form = reactive({
  username: 'admin',
  password: 'admin123',
  role: 'ADMIN'
})

async function submit() {
  loading.value = true
  try {
    await auth.login(form)
    ElMessage.success('登录成功')
    router.push('/dashboard')
  } catch (error) {
    ElMessage.error(error?.response?.data?.message || '登录失败')
  } finally {
    loading.value = false
  }
}
</script>
