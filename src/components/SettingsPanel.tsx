import { useRef } from 'react'
import { useAppStore } from '../store/useAppStore'
import Card from './Card'

export default function SettingsPanel() {
  const settings = useAppStore((s) => s.settings)
  const updateSettings = useAppStore((s) => s.updateSettings)
  const periods = useAppStore((s) => s.periods)
  const replaceAll = useAppStore((s) => s.replaceAll)
  const clearAll = useAppStore((s) => s.clearAll)
  const fileInputRef = useRef<HTMLInputElement>(null)

  function handleBackup() {
    const data = { periods, settings }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `par-level-planner-backup-${new Date().toISOString().slice(0, 10)}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  function handleRestoreClick() {
    fileInputRef.current?.click()
  }

  async function handleRestoreFile(file: File) {
    try {
      const text = await file.text()
      const data = JSON.parse(text)
      if (!Array.isArray(data.periods) || typeof data.settings !== 'object') {
        throw new Error('Invalid backup file')
      }
      if (confirm('This will replace all current data with the backup. Continue?')) {
        replaceAll(data)
      }
    } catch {
      alert('Could not read that file as a valid backup.')
    }
  }

  return (
    <div className="space-y-6">
      <Card title="Default buffer">
        <label className="block text-xs font-medium text-slate-400 mb-1">
          Default buffer % applied across the app
        </label>
        <input
          type="number"
          min={0}
          step={1}
          value={settings.defaultBufferPct}
          onChange={(e) => updateSettings({ defaultBufferPct: Number(e.target.value) || 0 })}
          className="w-32 rounded-md bg-slate-800 border border-slate-700 px-3 py-1.5 text-sm text-slate-100"
        />
        <p className="mt-2 text-xs text-slate-500">
          Padding added to recommended order quantities so you don't run out. You can still
          override it per-calculation in the Order Calculator tab.
        </p>
      </Card>

      <Card title="Data">
        <p className="text-sm text-slate-400 mb-4">
          All of your reports are stored only in this browser (nothing is sent to a server).
          Back up your data if you're clearing browser storage or switching devices.
        </p>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={handleBackup}
            className="rounded-md bg-slate-800 px-3 py-1.5 text-sm font-medium text-slate-200 hover:bg-slate-700"
          >
            Download backup (JSON)
          </button>
          <button
            onClick={handleRestoreClick}
            className="rounded-md bg-slate-800 px-3 py-1.5 text-sm font-medium text-slate-200 hover:bg-slate-700"
          >
            Restore from backup
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/json"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) void handleRestoreFile(file)
              e.target.value = ''
            }}
          />
          <button
            onClick={() => {
              if (confirm('This will permanently delete all saved reports. Continue?')) {
                clearAll()
              }
            }}
            className="rounded-md bg-red-950 px-3 py-1.5 text-sm font-medium text-red-300 hover:bg-red-900 ml-auto"
          >
            Clear all data
          </button>
        </div>
      </Card>
    </div>
  )
}
