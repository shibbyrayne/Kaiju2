import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { v4 as uuidv4 } from 'uuid'
import type { Period, Settings } from '../types'

interface AppState {
  periods: Period[]
  settings: Settings
  addPeriod: (period: Omit<Period, 'id' | 'createdAt'>) => void
  updatePeriod: (id: string, period: Omit<Period, 'id' | 'createdAt'>) => void
  removePeriod: (id: string) => void
  updateSettings: (settings: Partial<Settings>) => void
  replaceAll: (data: { periods: Period[]; settings: Settings }) => void
  clearAll: () => void
}

const defaultSettings: Settings = {
  defaultBufferPct: 15,
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      periods: [],
      settings: defaultSettings,
      addPeriod: (period) =>
        set((state) => ({
          periods: [
            ...state.periods,
            { ...period, id: uuidv4(), createdAt: new Date().toISOString() },
          ],
        })),
      updatePeriod: (id, period) =>
        set((state) => ({
          periods: state.periods.map((p) => (p.id === id ? { ...p, ...period } : p)),
        })),
      removePeriod: (id) =>
        set((state) => ({ periods: state.periods.filter((p) => p.id !== id) })),
      updateSettings: (settings) =>
        set((state) => ({ settings: { ...state.settings, ...settings } })),
      replaceAll: (data) => set({ periods: data.periods, settings: data.settings }),
      clearAll: () => set({ periods: [], settings: defaultSettings }),
    }),
    {
      name: 'par-level-planner-storage',
    },
  ),
)
