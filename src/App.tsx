import { useState } from 'react'
import ReportsPanel from './components/ReportsPanel'
import ItemAnalysisPanel from './components/ItemAnalysisPanel'
import OrderCalculatorPanel from './components/OrderCalculatorPanel'
import SettingsPanel from './components/SettingsPanel'

type Tab = 'reports' | 'analysis' | 'calculator' | 'settings'

const TABS: { id: Tab; label: string }[] = [
  { id: 'reports', label: 'Reports' },
  { id: 'analysis', label: 'Item Analysis' },
  { id: 'calculator', label: 'Order Calculator' },
  { id: 'settings', label: 'Settings' },
]

function App() {
  const [tab, setTab] = useState<Tab>('reports')

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800">
        <div className="max-w-5xl mx-auto px-4 py-5">
          <h1 className="text-xl font-bold tracking-tight">Par Level Planner</h1>
          <p className="text-sm text-slate-400 mt-1">
            Import your product mix and sales reports to see how many orders of each item you
            sell per $1,000 in sales — then get a buffered order recommendation.
          </p>
        </div>
      </header>

      <nav className="border-b border-slate-800 bg-slate-900/40">
        <div className="max-w-5xl mx-auto px-4 flex gap-1 overflow-x-auto">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-3 text-sm font-medium border-b-2 whitespace-nowrap transition-colors ${
                tab === t.id
                  ? 'border-emerald-500 text-emerald-400'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="max-w-5xl mx-auto px-4 py-6">
        {tab === 'reports' && <ReportsPanel />}
        {tab === 'analysis' && <ItemAnalysisPanel />}
        {tab === 'calculator' && <OrderCalculatorPanel />}
        {tab === 'settings' && <SettingsPanel />}
      </main>

      <footer className="max-w-5xl mx-auto px-4 py-6 text-xs text-slate-600">
        Data is stored locally in this browser only.
      </footer>
    </div>
  )
}

export default App
