import type { ItemStat, Period } from '../types'

export function normalizeItemName(name: string): string {
  return name.trim().toLowerCase().replace(/\s+/g, ' ')
}

/**
 * Aggregates saved periods into per-item stats. An item's ratio is a
 * sales-weighted average across every period it appeared in:
 *   (sum of units sold) / (sum of total sales of those periods / 1000)
 * Weighting by sales (rather than averaging each period's ratio equally)
 * keeps a single slow week from skewing the number as much as a
 * high-volume week.
 */
export function computeItemStats(periods: Period[], bufferPct: number): ItemStat[] {
  const byKey = new Map<
    string,
    {
      itemName: string
      category?: string
      totalQuantitySold: number
      totalSalesAcrossPeriods: number
      periodIds: Set<string>
      history: { periodId: string; label: string; date: string; ordersPer1000: number }[]
    }
  >()

  const sortedPeriods = [...periods].sort((a, b) => a.date.localeCompare(b.date))

  for (const period of sortedPeriods) {
    if (period.totalSales <= 0) continue

    // combine duplicate item rows within the same period
    const perPeriodQty = new Map<string, { itemName: string; category?: string; qty: number }>()
    for (const row of period.items) {
      const key = normalizeItemName(row.itemName)
      const existing = perPeriodQty.get(key)
      if (existing) {
        existing.qty += row.quantitySold
      } else {
        perPeriodQty.set(key, {
          itemName: row.itemName.trim(),
          category: row.category,
          qty: row.quantitySold,
        })
      }
    }

    for (const [key, { itemName, category, qty }] of perPeriodQty) {
      let entry = byKey.get(key)
      if (!entry) {
        entry = {
          itemName,
          category,
          totalQuantitySold: 0,
          totalSalesAcrossPeriods: 0,
          periodIds: new Set(),
          history: [],
        }
        byKey.set(key, entry)
      }
      entry.totalQuantitySold += qty
      entry.totalSalesAcrossPeriods += period.totalSales
      entry.periodIds.add(period.id)
      entry.history.push({
        periodId: period.id,
        label: period.label,
        date: period.date,
        ordersPer1000: (qty / period.totalSales) * 1000,
      })
    }
  }

  const stats: ItemStat[] = Array.from(byKey.values()).map((entry) => {
    const ordersPer1000 =
      entry.totalSalesAcrossPeriods > 0
        ? (entry.totalQuantitySold / entry.totalSalesAcrossPeriods) * 1000
        : 0
    return {
      itemName: entry.itemName,
      category: entry.category,
      totalQuantitySold: entry.totalQuantitySold,
      totalSalesAcrossPeriods: entry.totalSalesAcrossPeriods,
      periodCount: entry.periodIds.size,
      ordersPer1000,
      bufferedOrdersPer1000: ordersPer1000 * (1 + bufferPct / 100),
      history: entry.history,
    }
  })

  stats.sort((a, b) => b.ordersPer1000 - a.ordersPer1000)
  return stats
}

/** Recommended order quantity for a projected sales figure, buffer applied, rounded up. */
export function recommendedQuantity(
  ordersPer1000: number,
  projectedSales: number,
  bufferPct: number,
): number {
  const raw = ordersPer1000 * (projectedSales / 1000) * (1 + bufferPct / 100)
  return Math.ceil(raw)
}

export function sumItemsNetSales(items: { netSales?: number }[]): number | null {
  if (items.length === 0 || items.some((i) => i.netSales === undefined)) return null
  return items.reduce((sum, i) => sum + (i.netSales ?? 0), 0)
}
