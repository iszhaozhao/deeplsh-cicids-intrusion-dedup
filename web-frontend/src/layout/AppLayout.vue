<template>
  <div class="app-shell">
    <aside class="app-sidebar">
      <div class="brand">
        <div class="brand-mark">DH</div>
        <div>
          <div class="brand-title">日志去重平台</div>
          <div class="brand-subtitle">Deep Hashing Prototype</div>
        </div>
      </div>

      <el-menu
        class="app-menu"
        router
        :default-active="$route.path"
        background-color="#1e293b"
        text-color="#cbd5e1"
        active-text-color="#ffffff"
      >
        <el-menu-item index="/dashboard">首页</el-menu-item>
        <el-menu-item index="/logs/import">日志导入</el-menu-item>
        <el-menu-item index="/params">参数配置</el-menu-item>
        <el-menu-item index="/tasks">任务执行</el-menu-item>
        <el-menu-item index="/results">结果展示</el-menu-item>
        <el-menu-item index="/stats">统计分析</el-menu-item>
      </el-menu>
    </aside>

    <main class="app-main">
      <header class="app-header">
        <div>
          <div class="page-title">{{ pageTitle }}</div>
          <div class="page-subtitle">网络入侵检测日志去重轻量 Web 原型</div>
        </div>
        <div class="header-actions">
          <div class="user-badge">
            {{ auth.user?.realName || auth.user?.username }}
            <span>{{ auth.user?.role }}</span>
          </div>
          <el-button type="primary" plain @click="logout">退出登录</el-button>
        </div>
      </header>

      <section class="app-content">
        <router-view />
      </section>
    </main>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()

const titleMap = {
  '/dashboard': '系统首页',
  '/logs/import': '日志导入',
  '/params': '参数配置',
  '/tasks': '去重任务',
  '/results': '结果展示',
  '/stats': '统计分析'
}

const pageTitle = computed(() => titleMap[route.path] || '系统页面')

function logout() {
  auth.logout()
  router.push('/login')
}
</script>
