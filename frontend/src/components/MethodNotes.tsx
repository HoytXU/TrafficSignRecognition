import { useState } from 'react'
import { methodNotes } from '../data/projectData'

const COLOR_CLS = {
  amber:  { header: 'text-amber-400',   bg: 'bg-amber-500/5 border-amber-500/30',   dot: 'bg-amber-500' },
  blue:   { header: 'text-blue-400',    bg: 'bg-blue-500/5 border-blue-500/30',     dot: 'bg-blue-500'  },
  purple: { header: 'text-purple-400',  bg: 'bg-purple-500/5 border-purple-500/30', dot: 'bg-purple-500'},
} as const

export default function MethodNotes() {
  const [openId, setOpenId] = useState<string | null>(null)

  return (
    <section id="notes" className="py-20 px-4 border-t border-slate-800">
      <div className="max-w-5xl mx-auto">
        <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">Method Notes</h2>
        <p className="text-slate-500 text-sm mb-8">Expand each card for a concise explanation of the core technique.</p>

        <div className="flex flex-col gap-3">
          {methodNotes.map(note => {
            const c = COLOR_CLS[note.color]
            const isOpen = openId === note.id
            return (
              <div
                key={note.id}
                className={`rounded-xl border transition-colors ${isOpen ? c.bg : 'border-slate-700 bg-slate-800/40'}`}
              >
                <button
                  onClick={() => setOpenId(isOpen ? null : note.id)}
                  className="w-full flex items-center justify-between px-5 py-4 text-left"
                >
                  <span className={`text-sm font-semibold ${isOpen ? c.header : 'text-slate-200'}`}>
                    {note.title}
                  </span>
                  <span className="text-slate-500 text-lg leading-none ml-4 shrink-0">
                    {isOpen ? '−' : '+'}
                  </span>
                </button>

                {isOpen && (
                  <div className="px-5 pb-5">
                    <ul className="space-y-2">
                      {note.bullets.map((b, i) => (
                        <li key={i} className="flex gap-3 items-start">
                          <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${c.dot}`} />
                          <span className="text-sm text-slate-400 leading-relaxed">{b}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
