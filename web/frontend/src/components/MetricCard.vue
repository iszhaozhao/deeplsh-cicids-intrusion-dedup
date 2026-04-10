<template>
  <div class="metric-card">
    <div class="metric-label">{{ label }}</div>
    <div class="metric-value">{{ value }}</div>
    <div class="metric-hint">{{ hint }}</div>
  </div>
</template>

<script setup>
defineProps({
  label: String,
  value: [String, Number],
  hint: String
})
</script>
