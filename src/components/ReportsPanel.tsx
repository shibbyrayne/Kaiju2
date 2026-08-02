import { useState } from 'react'
import { useAppStore } from '../store/useAppStore'
import Card from './Card'
import ImportReportPanel from './ImportReportPanel'

export default function ReportsPanel() {
  const periods = useAppStore((s) => s.periods)
  const removePeriod = useAppStore((s) => s.removePeriod)
  const [importing, setImporting] = useState(false)

  const sorted = [...periods].sort((a, b) => b.date.localeCompare(a.date))

  return (
    <div className="space-y-6">
      <Card
        title="Import a report"
        action={
          !importing ? (
            <button
              onClick={() => setImporting(true)}
              className="rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500"
            >
              + Add report
            </button>
          ) : (
            <button
              onClick={() => setImporting(false)}
              className="rounded-md bg-slate-800 px-3 py-1.5 text-sm font-medium text-slate-300 hover:bg-slate-700"
            >
              Cancel
            </button>
          )
        }
      >
        {importing ? (
          <ImportReportPanel onDone={() => setImporting(false)} />
        ) : (
          <p className="text-sm text-slate-400">
            Import a product mix report (CSV) paired with the total sales for that same period.
            Add as many as you have — the more periods you load, the more reliable the
            orders-per-$1,000 ratio becomes.
          </p>
        )}
      </Card>

      <Card title={`Saved reports (${periods.length})`}>
        {sorted.length === 0 ? (
          <p className="text-sm text-slate-500">No reports yet. Import one above to get started.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-slate-400 border-b border-slate-800">
                  <th className="py-2 pr-4">Label</th>
                  <th className="py-2 pr-4">Date</th>
                  <th className="py-2 pr-4 text-right">Items</th>
                  <th className="py-2 pr-4 text-right">Total sales</th>
                  <th className="py-2 pr-4"></th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((p) => (
                  <tr key={p.id} className="border-b border-slate-900 text-slate-200">
                    <td className="py-2 pr-4">{p.label}</td>
                    <td className="py-2 pr-4 text-slate-400">{p.date}</td>
                    <td className="py-2 pr-4 text-right">{p.items.length}</td>
                    <td className="py-2 pr-4 text-right">
                      ${p.totalSales.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </td>
                    <td className="py-2 pr-4 text-right">
                      <button
                        onClick={() => {
                          if (confirm(`Remove "${p.label}"? This can't be undone.`)) {
                            removePeriod(p.id)
                          }
                        }}
                        className="text-xs text-red-400 hover:text-red-300"
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
