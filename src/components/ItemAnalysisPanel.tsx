import { useMemo, useState } from 'react'
import { useAppStore } from '../store/useAppStore'
import { computeItemStats } from '../lib/calc'
import { downloadCsv } from '../lib/exportCsv'
import Card from './Card'

export default function ItemAnalysisPanel() {
  const periods = useAppStore((s) => s.periods)
  const settings = useAppStore((s) => s.settings)
  const updateSettings = useAppStore((s) => s.updateSettings)

  const [search, setSearch] = useState('')
  const [bufferInput, setBufferInput] = useState(String(settings.defaultBufferPct))

  const bufferPct = Number(bufferInput) || 0
  const stats = useMemo(() => computeItemStats(periods, bufferPct), [periods, bufferPct])

  const filtered = stats.filter((s) => s.itemName.toLowerCase().includes(search.toLowerCase()))

  function handleExport() {
    downloadCsv(
      'item-analysis.csv',
      [
        'Item',
        'Category',
        'Total Qty Sold',
        'Total Sales ($)',
        'Periods',
        'Orders per $1,000',
        `Buffered (+${bufferPct}%)`,
      ],
      filtered.map((s) => [
        s.itemName,
        s.category ?? '',
        s.totalQuantitySold,
        s.totalSalesAcrossPeriods.toFixed(2),
        s.periodCount,
        s.ordersPer1000.toFixed(3),
        s.bufferedOrdersPer1000.toFixed(3),
      ]),
    )
  }

  return (
    <div className="space-y-6">
      <Card title="Item analysis">
        {periods.length === 0 ? (
          <p className="text-sm text-slate-500">
            Import at least one report to see item analysis.
          </p>
        ) : (
          <>
            <div className="flex flex-wrap items-end gap-4 mb-4">
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
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">
                  Buffer %
                </label>
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={bufferInput}
                  onChange={(e) => {
                    setBufferInput(e.target.value)
                    const n = Number(e.target.value)
                    if (!Number.isNaN(n)) updateSettings({ defaultBufferPct: n })
                  }}
                  className="w-24 rounded-md bg-slate-800 border border-slate-700 px-3 py-1.5 text-sm text-slate-100"
                />
              </div>
              <button
                onClick={handleExport}
                className="rounded-md bg-slate-800 px-3 py-1.5 text-sm font-medium text-slate-200 hover:bg-slate-700 ml-auto"
              >
                Export CSV
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-400 border-b border-slate-800">
                    <th className="py-2 pr-4">Item</th>
                    <th className="py-2 pr-4 text-right">Qty sold</th>
                    <th className="py-2 pr-4 text-right">Periods</th>
                    <th className="py-2 pr-4 text-right">Orders / $1,000</th>
                    <th className="py-2 pr-4 text-right">
                      Buffered ({bufferPct >= 0 ? '+' : ''}
                      {bufferPct}%)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((s) => (
                    <tr key={s.itemName} className="border-b border-slate-900 text-slate-200">
                      <td className="py-2 pr-4">
                        {s.itemName}
                        {s.category && (
                          <span className="ml-2 text-xs text-slate-500">{s.category}</span>
                        )}
                      </td>
                      <td className="py-2 pr-4 text-right">{s.totalQuantitySold}</td>
                      <td className="py-2 pr-4 text-right text-slate-400">{s.periodCount}</td>
                      <td className="py-2 pr-4 text-right font-medium">
                        {s.ordersPer1000.toFixed(2)}
                      </td>
                      <td className="py-2 pr-4 text-right font-medium text-emerald-400">
                        {s.bufferedOrdersPer1000.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {filtered.length === 0 && (
                <p className="text-sm text-slate-500 py-4">No items match your search.</p>
              )}
            </div>
          </>
        )}
      </Card>
    </div>
  )
}
