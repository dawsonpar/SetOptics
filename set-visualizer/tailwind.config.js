/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#e8f1fa', 100: '#c5ddf2', 200: '#9ec6e8', 300: '#74addd',
          400: '#4f99d4', 500: '#0761b2', 600: '#0656a0', 700: '#05478a',
          800: '#043972', 900: '#032b57',
        },
        surface: { DEFAULT: '#1a1a2e', light: '#252540', dark: '#0a0a0f' },
        muted: { DEFAULT: '#a0a0b8', dark: '#6b6b80' },
        accent: { DEFAULT: '#4f9cf7', dim: '#2d6bc4' },
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        mono: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        glow: '0 0 24px rgba(79, 156, 247, 0.35)',
        glass: '0 8px 32px rgba(7, 97, 178, 0.12), 0 2px 8px rgba(0,0,0,0.4)',
      },
    },
  },
  plugins: [],
}
