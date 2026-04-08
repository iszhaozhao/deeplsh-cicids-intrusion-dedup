import axios from 'axios'

const runtimeBaseUrl = (() => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL
  }
  if (typeof window !== 'undefined') {
    return window.location.hostname === '127.0.0.1'
      ? 'http://127.0.0.1:8080/api'
      : 'http://localhost:8080/api'
  }
  return 'http://127.0.0.1:8080/api'
})()

const http = axios.create({
  baseURL: runtimeBaseUrl,
  timeout: 20000
})

http.interceptors.request.use((config) => {
  const token = localStorage.getItem('demo-token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export default http
