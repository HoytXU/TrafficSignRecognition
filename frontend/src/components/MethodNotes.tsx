import { useState } from 'react'
import { methodNotes } from '../data/projectData'

export default function MethodNotes() {
  const [openId, setOpenId] = useState<string | null>(null)

  return (
    <section id="notes">
      <hr className="sec" />
      <h2 className="text-lg font-bold mb-4">6. Method Notes</h2>
      <p className="text-sm leading-relaxed mb-4">
        Brief technical notes on each core technique. Click a title to expand.
      </p>

      <div className="space-y-1">
        {methodNotes.map(note => {
          const isOpen = openId === note.id
          return (
            <div key={note.id} className="border border-gray-300">
              <button
                onClick={() => setOpenId(isOpen ? null : note.id)}
                className="w-full flex items-center justify-between px-4 py-2 text-left text-sm hover:bg-gray-50"
              >
                <strong>{note.title}</strong>
                <span className="text-gray-500 ml-4 shrink-0 font-mono">{isOpen ? '−' : '+'}</span>
              </button>
              {isOpen && (
                <div className="px-4 pb-4 border-t border-gray-200 bg-gray-50">
                  <ul className="text-sm list-disc list-inside space-y-1.5 mt-3 text-gray-700">
                    {note.bullets.map((b, i) => <li key={i}>{b}</li>)}
                  </ul>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}
