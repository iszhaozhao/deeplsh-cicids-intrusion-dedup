import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import LoginView from '../views/LoginView.vue'
import DashboardView from '../views/DashboardView.vue'
import LogImportView from '../views/LogImportView.vue'
import ParamsView from '../views/ParamsView.vue'
import TasksView from '../views/TasksView.vue'
import ResultsView from '../views/ResultsView.vue'
import StatsView from '../views/StatsView.vue'
import AppLayout from '../layout/AppLayout.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/login', component: LoginView },
    {
      path: '/',
      component: AppLayout,
      redirect: '/dashboard',
      meta: { requiresAuth: true },
      children: [
        { path: '/dashboard', component: DashboardView },
        { path: '/logs/import', component: LogImportView },
        { path: '/params', component: ParamsView },
        { path: '/tasks', component: TasksView },
        { path: '/results', component: ResultsView },
        { path: '/stats', component: StatsView }
      ]
    }
  ]
})

router.beforeEach((to) => {
  const auth = useAuthStore()
  if (to.meta.requiresAuth && !auth.isAuthenticated) {
    return '/login'
  }
  if (to.path === '/login' && auth.isAuthenticated) {
    return '/dashboard'
  }
  return true
})

export default router
