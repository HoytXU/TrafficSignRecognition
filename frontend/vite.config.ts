import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The base must match your GitHub repo name exactly.
// README links show: github.com/HoytXU/TrafficSignRecognition
// Update this if your repo name is different.
export default defineConfig({
  plugins: [react()],
  base: '/TrafficSignRecognition/',
  build: {
    outDir: '../docs',
    emptyOutDir: true,
  },
})
