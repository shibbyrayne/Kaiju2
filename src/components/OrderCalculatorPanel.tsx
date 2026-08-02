import { useMemo, useState } from 'react'
import { useAppStore } from '../store/useAppStore'
import { computeItemStats, recommendedQuantity } from '../lib/calc'
import { downloadCsv } from '../lib/exportCsv'
import Card from './Card'

export default function OrderCalculatorPanel() {
  const periods = useAppStore((s) => s.periods)
  const settings = useAppStore((s) => s.settings)

  const [projectedSales, setProjectedSales] = useState('')
  const [bufferInput, setBufferInput] = useState(String(settings.defaultBufferPct))
  const [search, setSearch] = useState('')

  const bufferPct = Number(bufferInput) || 0
  const projected = Number(projectedSales) || 0

  // base ratios never include the buffer twice — pull raw ordersPer1000 with buffer 0
  const stats = useMemo(() => computeItemStats(periods, 0), [periods])

  const rows = useMemo(
    () =>
      stats.map((s) => ({
        ...s,
        recommended: recommendedQuantity(s.ordersPer1000, projected, bufferPct),
      })),
    [stats, projected, bufferPct],
  )

  const filtered = rows.filter((r) => r.itemName.toLowerCase().includes(search.toLowerCase()))

  function handleExport() {
    downloadCsv(
      'order-recommendations.csv',
      ['Item', 'Orders per $1,000', `Projected sales ($${projected})`, `Buffer (${bufferPct}%)`, 'Recommended order qty'],
      filtered.map((r) => [
        r.itemName,
        r.ordersPer1000.toFixed(3),
        projected,
        bufferPct,
        r.recommended,
      ]),
    )
  }

  return (
    <div className="space-y-6">
      <Card title="Order calculator">
        {periods.length === 0 ? (
          <p className="text-sm text-slate-500">
            Import at least one report before calculating recommended order quantities.
          </p>
        ) : (
          <>
            <p className="text-sm text-slate-400 mb-4">
              Enter the sales figure you expect for your next order period (e.g. next week's
              projected sales), and a buffer to pad the result so you don't run out.
            </p>
            <div className="flex flex-wrap items-end gap-4 mb-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">
                  Projected sales ($)
                </label>
                <input
                  type="number"
                  min={0}
                  step={100}
                  value={projectedSales}
                  onChange={(e) => setProjectedSales(e.target.value)}
                  placeholder="e.g. 9000"
                  className="w-40 rounded-md bg-slate-800 border border-slate-700 px-3 py-1.5 text-sm text-slate-100"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">
                  Buffer % (optional)
                </label>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={bufferInput}
                  onChange={(e) => setBufferInput(e.target.value)}
                  className="w-24 rounded-md bg-slate-800 border border-slate-700 px-3 py-1.5 text-sm text-slate-100"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">
                  Search item
                </label>
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="e.g. Burger"
                  className="rounded-md bg-slate-800 border border-slate-700 px-3 py-1.5 text-sm text-slate-100"
                />
              </div>
              <button
                onClick={handleExport}
                disabled={projected <= 0}
                className="rounded-md bg-slate-800 px-3 py-1.5 text-sm font-medium text-slate-200 hover:bg-slate-700 ml-auto disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Export CSV
              </button>
            </div>

            {projected <= 0 ? (
              <p className="text-sm text-slate-500">
                Enter a projected sales amount above to see recommended order quantities.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-slate-400 border-b border-slate-800">
                      <th className="py-2 pr-4">Item</th>
                      <th className="py-2 pr-4 text-right">Orders / $1,000</th>
                      <th className="py-2 pr-4 text-right">
                        Recommended order qty
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((r) => (
                      <tr key={r.itemName} className="border-b border-slate-900 text-slate-200">
                        <td className="py-2 pr-4">{r.itemName}</td>
                        <td className="py-2 pr-4 text-right text-slate-400">
                          {r.ordersPer1000.toFixed(2)}
                        </td>
                        <td className="py-2 pr-4 text-right text-lg font-semibold text-emerald-400">
                          {r.recommended}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {filtered.length === 0 && (
                  <p className="text-sm text-slate-500 py-4">No items match your search.</p>
                )}
              </div>
            )}
          </>
        )}
      </Card>
    </div>
  )
}
