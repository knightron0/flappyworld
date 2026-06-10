import { defineConfig } from 'vite'

export default defineConfig({
  base: '/flappyworld/',
  assetsInclude: ['**/*.onnx'],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    fs: {
      allow: ['..'],
    },
  },
})
