import { defineConfig } from 'vite'

export default defineConfig({
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
