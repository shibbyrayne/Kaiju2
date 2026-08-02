/** Parses POS-report style numbers: "$1,234.50", "(12)" for negatives, "12%", stray whitespace. */
export function parseNumber(value: string | number | undefined | null): number {
  if (typeof value === 'number') return value
  if (!value) return 0
  let s = value.trim()
  if (s === '') return 0
  const isNegative = /^\(.*\)$/.test(s)
  s = s.replace(/[()$,%\s]/g, '')
  const n = parseFloat(s)
  if (Number.isNaN(n)) return 0
  return isNegative ? -n : n
}
