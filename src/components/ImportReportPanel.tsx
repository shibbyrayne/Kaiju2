import { useMemo, useState } from 'react'
import { parseCsvFile, type ParsedCsv } from '../lib/csv'
import { parseNumber } from '../lib/number'
import { sumItemsNetSales } from '../lib/calc'
import { useAppStore } from '../store/useAppStore'
import type { ProductMixRow } from '../types'

const NONE = '__none__'

function todayIso(): string {
  return new Date().toISOString().slice(0, 10)
}

function guessColumn(headers: string[], candidates: string[]): string {
  const lower = headers.map((h) => h.toLowerCase())
  for (const candidate of candidates) {
    const idx = lower.findIndex((h) => h.includes(candidate))
    if (idx !== -1) return headers[idx]
  }
  return NONE
}

export default function ImportReportPanel({ onDone }: { onDone: () => void }) {
  const addPeriod = useAppStore((s) => s.addPeriod)

  const [fileName, setFileName] = useState<string | null>(null)
  const [parsed, setParsed] = useState<ParsedCsv | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [itemNameCol, setItemNameCol] = useState(NONE)
  const [qtyCol, setQtyCol] = useState(NONE)
  const [categoryCol, setCategoryCol] = useState(NONE)
  const [netSalesCol, setNetSalesCol] = useState(NONE)

  const [label, setLabel] = useState('')
  const [date, setDate] = useState(todayIso())
  const [totalSalesMode, setTotalSalesMode] = useState<'sum' | 'manual'>('manual')
  const [manualTotalSales, setManualTotalSales] = useState('')

  async function handleFile(file: File) {
    setError(null)
    setFileName(file.name)
    try {
      const result = await parseCsvFile(file)
      setParsed(result)
      setItemNameCol(guessColumn(result.headers, ['item', 'menu', 'product', 'name']))
      setQtyCol(guessColumn(result.headers, ['qty', 'quantity', 'sold', 'count', '#']))
      setCategoryCol(guessColumn(result.headers, ['category', 'group', 'family']))
      setNetSalesCol(guessColumn(result.headers, ['net sales', 'sales', 'revenue', 'total']))
      if (!label) setLabel(file.name.replace(/\.[^.]+$/, ''))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to parse file')
      setParsed(null)
    }
  }

  const items: ProductMixRow[] = useMemo(() => {
    if (!parsed || itemNameCol === NONE || qtyCol === NONE) return []
    return parsed.rows
      .map((row) => ({
        itemName: (row[itemNameCol] ?? '').trim(),
        quantitySold: parseNumber(row[qtyCol]),
        category: categoryCol !== NONE ? row[categoryCol]?.trim() : undefined,
        netSales: netSalesCol !== NONE ? parseNumber(row[netSalesCol]) : undefined,
      }))
      .filter((r) => r.itemName.length > 0)
  }, [parsed, itemNameCol, qtyCol, categoryCol, netSalesCol])

  const summedNetSales = useMemo(() => sumItemsNetSales(items), [items])
  const canMap = parsed !== null
  const canSave =
    items.length > 0 &&
    label.trim().length > 0 &&
    ((totalSalesMode === 'sum' && summedNetSales !== null && summedNetSales > 0) ||
      (totalSalesMode === 'manual' && parseNumber(manualTotalSales) > 0))

  function handleSave() {
    const totalSales =
      totalSalesMode === 'sum' && summedNetSales !== null
        ? summedNetSales
        : parseNumber(manualTotalSales)
    addPeriod({
      label: label.trim(),
      date,
      totalSales,
      totalSalesSource: totalSalesMode === 'sum' ? 'summed-from-items' : 'manual',
      items,
    })
    onDone()
  }

  return (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-slate-200 mb-2">
          Product mix report (CSV)
        </label>
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) void handleFile(file)
          }}
          className="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:bg-emerald-600 file:text-white file:font-medium hover:file:bg-emerald-500 cursor-pointer"
        />
        <p className="mt-2 text-xs text-slate-500">
          Export your POS product mix report as CSV. If it's an Excel file, use "Save As &gt;
          CSV" first.
        </p>
        {fileName && <p className="mt-1 text-xs text-slate-400">Loaded: {fileName}</p>}
        {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
      </div>

      {canMap && parsed && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <ColumnSelect
              label="Item name column"
              value={itemNameCol}
              onChange={setItemNameCol}
              headers={parsed.headers}
              required
            />
            <ColumnSelect
              label="Quantity sold column"
              value={qtyCol}
              onChange={setQtyCol}
              headers={parsed.headers}
              required
            />
            <ColumnSelect
              label="Category column (optional)"
              value={categoryCol}
              onChange={setCategoryCol}
              headers={parsed.headers}
            />
            <ColumnSelect
              label="Net sales per item (optional)"
              value={netSalesCol}
              onChange={setNetSalesCol}
              headers={parsed.headers}
            />
          </div>

          {items.length > 0 && (
            <div>
              <p className="text-xs text-slate-500 mb-2">
                Parsed {items.length} item{items.length === 1 ? '' : 's'}. Preview:
              </p>
              <div className="max-h-40 overflow-auto rounded-md border border-slate-700">
                <table className="w-full text-xs">
                  <thead className="bg-slate-800 text-slate-400 sticky top-0">
                    <tr>
                      <th className="text-left px-2 py-1">Item</th>
                      <th className="text-right px-2 py-1">Qty</th>
                      {netSalesCol !== NONE && <th className="text-right px-2 py-1">Net Sales</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {items.slice(0, 8).map((it, i) => (
                      <tr key={i} className="border-t border-slate-800 text-slate-300">
                        <td className="px-2 py-1">{it.itemName}</td>
                        <td className="px-2 py-1 text-right">{it.quantitySold}</td>
                        {netSalesCol !== NONE && (
                          <td className="px-2 py-1 text-right">${it.netSales?.toFixed(2)}</td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-200 mb-1">
                Period label
              </label>
              <input
                type="text"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="e.g. Week of Jul 21"
                className="w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-slate-100"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-200 mb-1">Date</label>
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-slate-100"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-200 mb-2">
              Total sales for this period
            </label>
            <div className="flex flex-col gap-2">
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input
                  type="radio"
                  checked={totalSalesMode === 'sum'}
                  disabled={summedNetSales === null}
                  onChange={() => setTotalSalesMode('sum')}
                />
                Sum the net sales column
                {summedNetSales !== null ? (
                  <span className="text-slate-400">
                    (${summedNetSales.toLocaleString(undefined, { maximumFractionDigits: 2 })})
                  </span>
                ) : (
                  <span className="text-slate-500">(map a net sales column above)</span>
                )}
              </label>
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input
                  type="radio"
                  checked={totalSalesMode === 'manual'}
                  onChange={() => setTotalSalesMode('manual')}
                />
                Enter total sales manually
              </label>
              {totalSalesMode === 'manual' && (
                <input
                  type="text"
                  inputMode="decimal"
                  value={manualTotalSales}
                  onChange={(e) => setManualTotalSales(e.target.value)}
                  placeholder="e.g. 8500"
                  className="w-full max-w-xs rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-slate-100"
                />
              )}
            </div>
            <p className="mt-2 text-xs text-slate-500">
              This is total net sales for the whole restaurant over the same period as the
              product mix report — from your sales summary report, or typed in by hand.
            </p>
          </div>

          <button
            onClick={handleSave}
            disabled={!canSave}
            className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Save report
          </button>
        </>
      )}
    </div>
  )
}

function ColumnSelect({
  label,
  value,
  onChange,
  headers,
  required,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  headers: string[]
  required?: boolean
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-200 mb-1">
        {label} {required && <span className="text-red-400">*</span>}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-slate-100"
      >
        <option value={NONE}>{required ? 'Select a column…' : 'None'}</option>
        {headers.map((h) => (
          <option key={h} value={h}>
            {h}
          </option>
        ))}
      </select>
    </div>
  )
}
