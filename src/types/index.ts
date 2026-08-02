export interface ProductMixRow {
  itemName: string
  category?: string
  quantitySold: number
  /** Net sales for this item, if the report included it. Optional. */
  netSales?: number
}

export interface Period {
  id: string
  /** Short human label, e.g. "Week of Jul 21" or "July 2026" */
  label: string
  /** ISO date string for the period (used for sorting/trend charts) */
  date: string
  /** Total net sales for the whole period, in dollars */
  totalSales: number
  /** Where totalSales came from, purely informational */
  totalSalesSource: 'manual' | 'summed-from-items'
  items: ProductMixRow[]
  createdAt: string
}

export interface Settings {
  /** Default buffer percentage applied to recommended order quantities, e.g. 15 = +15% */
  defaultBufferPct: number
}

export interface ItemStat {
  itemName: string
  category?: string
  /** Total units sold for this item across all included periods */
  totalQuantitySold: number
  /** Total sales across all included periods that contained this item's data */
  totalSalesAcrossPeriods: number
  /** How many saved periods this item appeared in */
  periodCount: number
  /** Raw ratio: units sold per $1,000 in total sales, weighted across periods */
  ordersPer1000: number
  /** ordersPer1000 with the buffer percentage applied */
  bufferedOrdersPer1000: number
  /** History of per-period ratios, oldest first, for trend display */
  history: { periodId: string; label: string; date: string; ordersPer1000: number }[]
}
