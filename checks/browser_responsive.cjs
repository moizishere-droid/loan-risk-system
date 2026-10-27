/* Real Streamlit browser checks against checks.browser_service fixture predictions.
   Usage: NODE_PATH=<playwright packages> node checks/browser_responsive.cjs */
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const output = path.resolve('logs/ui-check');
fs.mkdirSync(output, { recursive: true });

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    await page.goto('http://127.0.0.1:17860');
    await page.getByRole('button', { name: 'Get started →', exact: true }).waitFor();
    async function check(name, width) {
      await page.setViewportSize({ width, height: 1000 });
      await page.waitForTimeout(300);
      const issues = await page.evaluate(() => {
        const main = document.querySelector('[data-testid="stAppViewContainer"] .main');
        const overflow = document.documentElement.scrollWidth > innerWidth + 2;
        const controls = [...document.querySelectorAll('.main input, .main button, .main [data-baseweb="select"]')]
          .filter(el => el.getBoundingClientRect().width && el.getBoundingClientRect().height &&
                        el.checkVisibility({ opacityProperty: true, visibilityProperty: true }))
          .filter(el => {
            const r = el.getBoundingClientRect();
            for (let parent = el.parentElement; parent; parent = parent.parentElement) {
              if (['hidden', 'clip', 'auto', 'scroll'].includes(getComputedStyle(parent).overflowX)) {
                const clip = parent.getBoundingClientRect();
                if (r.right <= clip.left || r.left >= clip.right) return false;
              }
            }
            return true;
          })
          .filter(el => { const r = el.getBoundingClientRect(); return r.left < -2 || r.right > innerWidth + 2; })
          .map(el => ({ text: el.textContent || el.getAttribute('aria-label'), rect: el.getBoundingClientRect().toJSON() }));
        return { overflow, controls, mainWidth: main?.getBoundingClientRect().width };
      });
      assert.equal(issues.overflow, false, `${name}/${width}: page overflows`);
      assert.deepEqual(issues.controls, [], `${name}/${width}: clipped controls: ${JSON.stringify(issues.controls)}`);
      assert.equal(await page.getByTestId('stException').count(), 0, `${name}: Streamlit exception`);
      await page.screenshot({ path: path.join(output, `${name}-${width}.png`), fullPage: true });
      console.log(`${name} ${width}px: PASS`);
    }
    async function sizes(name) {
      for (const width of [1440, 1024, 768, 390, 320]) await check(name, width);
      await page.setViewportSize({ width: 1440, height: 1000 });
      await page.waitForTimeout(700);
    }
    async function workflow(name) {
      const label = page.getByTestId('stSidebar').getByRole('button', { name, exact: true });
      const bounds = await label.boundingBox();
      if (!bounds || bounds.x < 0) {
        await page.getByTestId('collapsedControl').click();
        await page.waitForTimeout(700);
      }
      await label.click();
      await page.getByRole('heading', { name: name.charAt(0) + name.slice(1).toLowerCase(), exact: true }).waitFor();
    }
    await sizes('landing');
    await page.getByRole('button', { name: 'Get started →', exact: true }).click();
    await page.getByLabel('Full name', { exact: true }).waitFor();
    await sizes('signup');
    await page.getByLabel('Full name', { exact: true }).fill('Browser Check');
    await page.getByLabel('Email address', { exact: true }).fill(`browser-${Date.now()}@example.com`);
    await page.getByLabel('Password', { exact: true }).fill('browser-password-123');
    await page.getByLabel('Confirm password', { exact: true }).fill('browser-password-123');
    await page.getByRole('button', { name: 'Create account →', exact: true }).click();
    await page.getByRole('heading', { name: 'New applicant', exact: true }).waitFor();
    await sizes('new-applicant');
    await page.getByLabel('CNIC', { exact: true }).fill('42101-1234567-1');
    await page.getByRole('button', { name: 'Run assessment →', exact: true }).click();
    await page.getByRole('heading', { name: 'Assessment results', exact: true }).waitFor();
    await sizes('results');
    await workflow('Existing Loan');
    await page.getByLabel('Assigned interest rate (%)', { exact: true }).waitFor();
    await sizes('existing-loan');
    await workflow('Applicant Profile');
    await page.getByLabel('Applicant CNIC', { exact: true }).fill('42101-1234567-1');
    await page.getByRole('button', { name: 'Find applicant', exact: true }).click();
    await page.getByRole('heading', { name: 'Applicant summary', exact: true }).waitFor();
    await sizes('profile');
    await workflow('Delete Applicant');
    await page.getByLabel('Applicant CNIC', { exact: true }).fill('42101-1234567-1');
    await page.getByRole('button', { name: 'Review records', exact: true }).click();
    await page.getByLabel('Type DELETE to confirm', { exact: true }).waitFor();
    await sizes('delete-review');
    await page.getByRole('button', { name: '◐', exact: true }).click();
    await check('dark-delete', 1440);
    await page.getByRole('button', { name: 'Log out', exact: true }).click();
    await page.getByRole('heading', { name: 'Welcome back', exact: true }).waitFor();
    await sizes('login');
    await check('dark-login', 390);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
