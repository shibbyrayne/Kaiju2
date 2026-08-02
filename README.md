# Par Level Planner

A small app for figuring out how many orders of a menu item you sell per
$1,000 in sales — so you can size purchase/order quantities off projected
sales instead of guessing.

## How it works

1. **Reports tab** — Import a product mix report (CSV export from your POS)
   along with the total net sales for that same period. You map which
   columns are the item name, quantity sold, category (optional), and
   per-item net sales (optional). If your report doesn't total sales,
   type the number in by hand from your sales summary. Add as many periods
   as you have — more history makes the ratio more reliable.
2. **Item Analysis tab** — Every saved period is combined into a
   sales-weighted ratio per item: `units sold / (total sales / 1000)`. A
   buffer percentage (set here or in Settings) pads the number so you don't
   cut it too close.
3. **Order Calculator tab** — Enter a projected sales figure for your next
   order period (and optionally override the buffer), and get a
   recommended order quantity per item, ready to export as CSV.
4. **Settings tab** — Set your default buffer %, back up all data to a JSON
   file, restore from a backup, or clear everything.

All data is stored only in the browser's local storage — nothing is sent
to a server. Back up from Settings before clearing your browser data or
moving to a different computer/browser.

CSV only: if your POS exports Excel, use "Save As → CSV" first.

## Development

```bash
npm install
npm run dev      # start local dev server
npm run build    # type-check and build for production into dist/
npm run preview  # preview the production build
npm run lint      # oxlint
```

## Deploying for restaurant use

Since it's a self-contained static site, the simplest options are:

- Run `npm run build` and open `dist/index.html` directly in a browser, or
  serve the `dist/` folder with any static file server.
- Host `dist/` for free on GitHub Pages, Netlify, Vercel, etc., and bookmark
  it on the computer/tablet you use in the restaurant.

Because data lives in that browser's local storage, use the same
browser/device each time (or export/import a backup) to keep your report
history.
