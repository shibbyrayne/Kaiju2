import Papa from 'papaparse'

export interface ParsedCsv {
  headers: string[]
  rows: Record<string, string>[]
}

export function parseCsvFile(file: File): Promise<ParsedCsv> {
  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: 'greedy',
      complete: (results) => {
        const headers = results.meta.fields ?? []
        if (headers.length === 0) {
          reject(new Error('No columns found in this file. Is it a valid CSV?'))
          return
        }
        resolve({ headers, rows: results.data })
      },
      error: (err: Error) => reject(err),
    })
  })
}
