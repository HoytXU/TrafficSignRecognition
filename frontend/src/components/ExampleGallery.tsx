import { useState } from 'react'
import { galleryItems, type GalleryItem } from '../data/projectData'

function Modal({ item, onClose }: { item: GalleryItem; onClose: () => void }) {
  const base = import.meta.env.BASE_URL
  return (
    <div
      className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-slate-800 rounded-xl border border-slate-700 max-w-sm w-full p-5"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-white">Class {item.classId} — GTSRB</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white text-lg leading-none">×</button>
        </div>

        <img
          src={`${base}${item.imagePath}`}
          alt={`GTSRB class ${item.classId} reference`}
          className="w-full rounded-lg mb-4 bg-slate-700"
        />

        <div className="flex items-center gap-2 mb-4">
          <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20">
            Placeholder predictions
          </span>
          <span className="text-xs text-slate-500">No inference was run</span>
        </div>

        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">True label</span>
            <span className="text-emerald-400">{item.trueLabel}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Predicted</span>
            <span className="text-blue-300">{item.predictedLabel}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Confidence</span>
            <span className="text-slate-200">{(item.confidence * 100).toFixed(1)}%</span>
          </div>
        </div>

        <div className="mt-4">
          <div className="text-xs text-slate-500 mb-2">Top-5 predictions</div>
          <div className="space-y-1">
            {item.topFive.map((t, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-xs text-slate-500 w-4">{i + 1}.</span>
                <div className="flex-1 bg-slate-700 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full ${i === 0 ? 'bg-blue-500' : 'bg-slate-500'}`}
                    style={{ width: `${(t.prob * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="text-xs text-slate-400 w-16 text-right">
                  Class {t.classId} {(t.prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        <p className="mt-4 text-xs text-slate-600 leading-relaxed">
          These are the GTSRB class reference images. Displayed predictions are hardcoded placeholder values — connect a model backend to show real inference results.
        </p>
      </div>
    </div>
  )
}

export default function ExampleGallery() {
  const [selected, setSelected] = useState<GalleryItem | null>(null)
  const [showAll, setShowAll] = useState(false)
  const base = import.meta.env.BASE_URL

  const displayed = showAll ? galleryItems : galleryItems.slice(0, 12)

  return (
    <section id="gallery" className="py-20 px-4 border-t border-slate-800">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-start justify-between flex-wrap gap-4 mb-2">
          <div>
            <h2 className="text-2xl md:text-3xl font-bold text-white">Example Gallery</h2>
            <p className="text-slate-500 text-sm mt-1">
              GTSRB class reference images (classes 0–42). Click any image to inspect.
            </p>
          </div>
          <span className="text-xs px-2 py-1 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/20 self-start mt-1">
            Placeholder predictions
          </span>
        </div>

        <p className="text-xs text-slate-600 mb-8">
          These are the per-class reference images from the GTSRB Meta/ directory. Confidence scores and top-5 predictions shown in the modal are hardcoded placeholders — no model inference was run.
        </p>

        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-12 gap-2 mb-6">
          {displayed.map(item => (
            <button
              key={item.id}
              onClick={() => setSelected(item)}
              className="group relative aspect-square rounded-lg overflow-hidden border border-slate-700 hover:border-blue-500 transition-colors bg-slate-800"
              title={`Class ${item.classId}`}
            >
              <img
                src={`${base}${item.imagePath}`}
                alt={`Class ${item.classId}`}
                className="w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <span className="text-xs text-white font-medium">{item.classId}</span>
              </div>
            </button>
          ))}
        </div>

        {!showAll && (
          <button
            onClick={() => setShowAll(true)}
            className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            Show all 43 classes ↓
          </button>
        )}

        {selected && <Modal item={selected} onClose={() => setSelected(null)} />}
      </div>
    </section>
  )
}
