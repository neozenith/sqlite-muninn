import { defineConfig, devices } from '@playwright/test'

/**
 * Playwright config for muninn-viz.
 *
 * - Backend (FastAPI) at :8200, frontend (Vite) at :5280
 * - All artifacts (screenshots, .log, .network.json, failure traces) land in
 *   e2e-screenshots/ so the whole post-mortem set lives in one folder.
 * - globalSetup warms Vite so the first test in each worker doesn't pay the
 *   lazy-compile tax.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['list'],
    ['html', { outputFolder: 'playwright-report', open: 'never' }],
  ],
  outputDir: 'e2e-screenshots',
  timeout: 150000,
  expect: { timeout: 10000 },
  globalSetup: './e2e/global-setup.ts',

  use: {
    baseURL: 'http://localhost:5282',
    viewport: { width: 1280, height: 720 },
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        // WebGL args for Deck.GL to initialize in headless Chromium.
        // `use-gl=angle` + `angle=swiftshader` gives us a working software
        // rasterizer; `ignore-gpu-blocklist` skips the sandbox-based refusal
        // that headless chrome otherwise applies on Linux CI.
        launchOptions: {
          args: [
            '--enable-webgl',
            '--use-gl=angle',
            '--use-angle=swiftshader',
            '--ignore-gpu-blocklist',
          ],
        },
      },
    },
  ],

  webServer: {
    command:
      'concurrently --kill-others --names "be,fe" ' +
      '"uv run --directory .. python -m server --port 8200" ' +
      '"VITE_API_PORT=8200 npx vite --port 5282 --strictPort"',
    url: 'http://localhost:5282',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },
})
