import { expect, test } from '@playwright/test'

test('homepage renders the bare scaffold', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'muninn-viz' })).toBeVisible()
})
