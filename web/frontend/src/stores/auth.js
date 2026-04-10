import { defineStore } from 'pinia'
import http from '../api/http'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: localStorage.getItem('demo-token') || '',
    user: JSON.parse(localStorage.getItem('demo-user') || 'null')
  }),
  getters: {
    isAuthenticated: (state) => Boolean(state.token)
  },
  actions: {
    async login(payload) {
      const { data } = await http.post('/auth/login', payload)
      this.token = data.token
      this.user = {
        username: data.username,
        realName: data.realName,
        role: data.role
      }
      localStorage.setItem('demo-token', this.token)
      localStorage.setItem('demo-user', JSON.stringify(this.user))
    },
    async fetchMe() {
      if (!this.token) return null
      const { data } = await http.get('/auth/me')
      this.user = data
      localStorage.setItem('demo-user', JSON.stringify(this.user))
      return data
    },
    logout() {
      this.token = ''
      this.user = null
      localStorage.removeItem('demo-token')
      localStorage.removeItem('demo-user')
    }
  }
})
