import { defineConfig } from 'vite'
import path from 'node:path'
import electron from 'vite-plugin-electron/simple'
import react from '@vitejs/plugin-react'
import { annotationWebApi } from './web/fileApi'


// Browser mode (ANNOTATION_WEB=1): no Electron, file access goes through web/fileApi.ts.
const isWeb = process.env.ANNOTATION_WEB === '1'
const setOpticsRoot = path.resolve(__dirname, '..', '..')

function webPlugins() {
  const token = process.env.ANNOTATION_WEB_TOKEN
  if (!token || token.length < 16) throw new Error('ANNOTATION_WEB_TOKEN of at least 16 characters is required in browser mode')
  return [annotationWebApi({
    setOpticsRoot,
    videoRoot: process.env.ANNOTATION_VIDEO_ROOT ?? path.join(setOpticsRoot, 'data', 'rally-gt'),
    token,
  })]
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    ...(isWeb ? webPlugins() : [electron({
      main: {
        // Shortcut of `build.lib.entry`.
        entry: 'electron/main.ts',
        vite: {
          build: {

          }
        }
      },
      preload: {
        // Shortcut of `build.rollupOptions.input`.
        // Preload scripts may contain Web assets, so use the `build.rollupOptions.input` instead `build.lib.entry`.
        input: path.join(__dirname, 'electron/preload.ts'),
      },
      // Ployfill the Electron and Node.js API for Renderer process.
      // If you want use Node.js in Renderer process, the `nodeIntegration` needs to be enabled in the Main process.
      // See https://github.com/electron-vite/vite-plugin-electron-renderer
      renderer: process.env.NODE_ENV === 'test'
        // https://github.com/electron-vite/vite-plugin-electron-renderer/issues/78#issuecomment-2053600808
        ? undefined
        : {},
    })]),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
    },
  },
  build: {
    target: 'esnext',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true,
        pure_funcs: ['console.log', 'console.debug']
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          'pixi': ['pixi.js'],
          'react-vendor': ['react', 'react-dom'],
          'video-processing': ['mediabunny', 'mp4box', '@fix-webm-duration/fix']
        }
      }
    },
    chunkSizeWarningLimit: 1000
  }
})
