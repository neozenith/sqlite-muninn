import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:5281',
    colorScheme: 'dark',
    video: { mode: 'on', size: { width: 1280, height: 720 } },
    viewport: { width: 1280, height: 720 },
    screenshot: 'off', // We handle screenshots manually via checkpoint()
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
    },
  ],
  webServer: {
    command:
      'VITE_API_PORT=8201 npx concurrently --kill-others ' +
      '"uv run --directory .. python -m server --port 8201" ' +
      '"npx vite --port 5281"',
    port: 5281,
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
