/**
 * E2E test checkpoint helper.
 * Each checkpoint takes a screenshot and asserts zero unexpected console errors.
 */
import { Page, expect } from '@playwright/test';

/** Known benign errors from libraries and background fetches in headless mode. */
const BENIGN_PATTERNS = [
  'luma',         // Deck.GL luma.gl WebGL errors in headless Chromium
  'WebGL',        // WebGL context creation failures
  'deck:',        // Deck.GL internal warnings
  'Failed to initialize WebGL',
  'Failed to load resource',  // Chrome generic 500 for background API fetches (logged via failedResponses)
  'maxTextureDimension',  // luma.gl headless: no GPU adapter → undefined device limits
];

function isBenign(msg: string): boolean {
  return BENIGN_PATTERNS.some((p) => msg.includes(p));
}

let consoleErrors: string[] = [];
let failedResponses: string[] = [];

/** Attach a console error listener and response tracker. Call once per test. */
export function setupConsoleMonitor(page: Page) {
  consoleErrors = [];
  failedResponses = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error' && !isBenign(msg.text())) {
      consoleErrors.push(msg.text());
    }
  });
  page.on('response', (response) => {
    if (response.status() >= 500) {
      failedResponses.push(`${response.status()} ${response.url()}`);
    }
  });
}

/** Take a screenshot and assert no unexpected console errors since last checkpoint. */
export async function checkpoint(page: Page, id: string) {
  if (consoleErrors.length > 0) {
    console.log(`[checkpoint:${id}] Console errors:`, consoleErrors);
    console.log(`[checkpoint:${id}] Failed responses:`, failedResponses);
  }
  expect(consoleErrors, `Console errors at checkpoint ${id}`).toEqual([]);
  await page.screenshot({ path: `screenshots/${id}.png`, fullPage: true });
}
