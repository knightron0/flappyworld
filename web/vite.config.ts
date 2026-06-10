import { defineConfig } from 'vite'

export default defineConfig({
  base: '/',
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
