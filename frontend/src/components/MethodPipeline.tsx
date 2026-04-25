import { useState } from 'react'
import { pipelineMethods } from '../data/projectData'

export default function MethodPipeline() {
  const [activeId, setActiveId] = useState(pipelineMethods[0].id)
  const active = pipelineMethods.find(m => m.id === activeId)!

  return (
    <section id="methods">
      <hr className="sec" />
      <h2 className="text-lg font-bold mb-4">2. Method Pipelines</h2>

      <p className="text-sm leading-relaxed mb-4">
        Each of the three levels follows a common pipeline structure:
        input image → preprocessing → feature extraction → classifier → prediction.
        The figure below shows the specific instantiation for each approach.
      </p>

      {/* Tab selector */}
      <p className="text-sm mb-3">
        <span className="text-gray-500">Select method: </span>
        {pipelineMethods.map((m, i) => (
          <span key={m.id}>
            {i > 0 && <span className="text-gray-400 mx-1">·</span>}
            <button
              onClick={() => setActiveId(m.id)}
              className={`text-sm ${
                activeId === m.id
                  ? 'font-bold underline text-black'
                  : 'text-blue-700 hover:underline'
              }`}
            >
              {m.name}
            </button>
          </span>
        ))}
      </p>

      {/* Pipeline diagram */}
      <div className="overflow-x-auto pipeline-scroll pb-2 mb-3 border border-gray-300 rounded p-3 bg-gray-50">
        <div className="flex items-start gap-1 min-w-max">
          {active.steps.map((step, i) => (
            <div key={step.label} className="flex items-start gap-1">
              <div className="border border-gray-400 bg-white px-3 py-2 text-center" style={{ minWidth: 110 }}>
                <div className="text-xs font-bold">{step.label}</div>
                <div className="text-xs text-gray-600 mt-1 leading-tight">{step.detail}</div>
              </div>
              {i < active.steps.length - 1 && (
                <span className="text-gray-500 text-sm mt-2 shrink-0">→</span>
              )}
            </div>
          ))}
        </div>
      </div>
      <p className="text-xs text-gray-500 italic mb-4">
        Figure 1. Pipeline for <em>{active.name}</em>.
      </p>

      {/* Highlights as bullet list */}
      <ul className="text-sm list-disc list-inside text-gray-700 space-y-1">
        {active.highlights.map(h => <li key={h}>{h}</li>)}
      </ul>
    </section>
  )
}
