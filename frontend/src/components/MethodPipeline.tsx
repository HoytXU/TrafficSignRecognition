import { useState } from 'react'
import { pipelineMethods } from '../data/projectData'

const colorCls = {
  amber:   { tab: 'border-amber-500 text-amber-400',   tabInactive: 'text-slate-400 hover:text-amber-300 border-transparent', step: 'border-amber-500/40 bg-amber-500/5', label: 'text-amber-400', badge: 'bg-amber-500/10 text-amber-300' },
  emerald: { tab: 'border-emerald-500 text-emerald-400', tabInactive: 'text-slate-400 hover:text-emerald-300 border-transparent', step: 'border-emerald-500/40 bg-emerald-500/5', label: 'text-emerald-400', badge: 'bg-emerald-500/10 text-emerald-300' },
  blue:    { tab: 'border-blue-500 text-blue-400',     tabInactive: 'text-slate-400 hover:text-blue-300 border-transparent', step: 'border-blue-500/40 bg-blue-500/5', label: 'text-blue-400', badge: 'bg-blue-500/10 text-blue-300' },
  purple:  { tab: 'border-purple-500 text-purple-400', tabInactive: 'text-slate-400 hover:text-purple-300 border-transparent', step: 'border-purple-500/40 bg-purple-500/5', label: 'text-purple-400', badge: 'bg-purple-500/10 text-purple-300' },
} as const

export default function MethodPipeline() {
  const [activeId, setActiveId] = useState(pipelineMethods[0].id)
  const active = pipelineMethods.find(m => m.id === activeId)!
  const c = colorCls[active.color]

  return (
    <section id="methods" className="py-20 px-4 border-t border-slate-800">
      <div className="max-w-5xl mx-auto">
        <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">Method Pipeline Explorer</h2>
        <p className="text-slate-500 text-sm mb-8">Input → preprocessing → feature extraction → classifier → prediction</p>

        {/* Tab bar */}
        <div className="flex gap-1 border-b border-slate-700 mb-8 overflow-x-auto">
          {pipelineMethods.map(m => {
            const isActive = m.id === activeId
            const tc = colorCls[m.color]
            return (
              <button
                key={m.id}
                onClick={() => setActiveId(m.id)}
                className={`px-4 py-2 text-sm font-medium border-b-2 whitespace-nowrap transition-colors -mb-px ${
                  isActive ? tc.tab : tc.tabInactive
                }`}
              >
                {m.name}
              </button>
            )
          })}
        </div>

        {/* Pipeline steps */}
        <div className="overflow-x-auto pipeline-scroll pb-2 mb-8">
          <div className="flex items-center gap-2 min-w-max">
            {active.steps.map((step, i) => (
              <div key={step.label} className="flex items-center gap-2">
                <div className={`rounded-lg border ${c.step} px-4 py-3 min-w-[130px]`}>
                  <div className={`text-xs font-semibold uppercase tracking-wide mb-1 ${c.label}`}>
                    {step.label}
                  </div>
                  <div className="text-xs text-slate-400 leading-relaxed">{step.detail}</div>
                </div>
                {i < active.steps.length - 1 && (
                  <span className="text-slate-600 text-lg shrink-0">→</span>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Highlights */}
        <div className="flex flex-wrap gap-2">
          {active.highlights.map(h => (
            <span key={h} className={`text-xs px-3 py-1 rounded-full ${c.badge}`}>
              {h}
            </span>
          ))}
        </div>
      </div>
    </section>
  )
}
