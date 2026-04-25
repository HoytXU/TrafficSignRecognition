import { failureCases } from '../data/projectData'

export default function FailureCases() {
  const base = import.meta.env.BASE_URL

  return (
    <section id="failures" className="py-20 px-4 border-t border-slate-800">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-start justify-between flex-wrap gap-4 mb-2">
          <div>
            <h2 className="text-2xl md:text-3xl font-bold text-white">Failure Cases</h2>
            <p className="text-slate-500 text-sm mt-1">
              Common patterns where traffic sign classifiers go wrong.
            </p>
          </div>
          <span className="text-xs px-2 py-1 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20 self-start mt-1">
            Placeholder examples
          </span>
        </div>

        <p className="text-xs text-slate-600 mb-8">
          These are illustrative failure modes constructed from class reference images. Images and misclassification labels are placeholders — they represent real failure patterns but do not come from saved test-set predictions.
        </p>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {failureCases.map((fc, i) => (
            <div key={i} className="bg-slate-800 rounded-xl border border-slate-700 p-4">
              <div className="flex gap-4 items-start mb-3">
                <img
                  src={`${base}${fc.imagePath}`}
                  alt="Failure case input"
                  className="w-16 h-16 object-cover rounded-lg bg-slate-700 shrink-0"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1 mb-1">
                    <span className="text-xs text-slate-500">Predicted</span>
                    <span className="text-xs text-red-400 font-medium">{fc.predictedLabel}</span>
                    <span className="text-xs text-slate-600">({(fc.confidence * 100).toFixed(0)}%)</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-slate-500">True</span>
                    <span className="text-xs text-emerald-400 font-medium">{fc.trueLabel}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-1 mb-2">
                <div className="w-full bg-slate-700 rounded-full h-1">
                  <div
                    className="bg-red-500 h-1 rounded-full"
                    style={{ width: `${(fc.confidence * 100).toFixed(0)}%` }}
                  />
                </div>
                <span className="text-xs text-red-400 shrink-0">{(fc.confidence * 100).toFixed(0)}%</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">{fc.reason}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
