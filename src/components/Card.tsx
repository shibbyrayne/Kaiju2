import type { ReactNode } from 'react'

export default function Card({
  title,
  action,
  children,
}: {
  title?: string
  action?: ReactNode
  children: ReactNode
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-5">
      {(title || action) && (
        <div className="flex items-center justify-between mb-4">
          {title && <h2 className="text-base font-semibold text-slate-100">{title}</h2>}
          {action}
        </div>
      )}
      {children}
    </div>
  )
}
