import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
  server:{
    proxy:{
      //匹配所有以/api开头的请求（和后端接口路径对应
      '/api':{
        target:'https://ddtwk.site',
        changeOrigin:true,//是否开启本地代理
        rewrite:(path)=>path,//重写路径
        // secure:false
      }
    }
  },

})
