import { useState } from 'react'
import { galleryItems, getGtsrbClassLabel, type GalleryItem } from '../data/projectData'

function Modal({ item, onClose }: { item: GalleryItem; onClose: () => void }) {
  const base = import.meta.env.BASE_URL
  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white border border-gray-400 max-w-sm w-full p-5"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <strong className="text-sm">Class {item.classId}: {item.trueLabel}</strong>
          <button onClick={onClose} className="text-gray-400 hover:text-black text-lg leading-none">×</button>
        </div>

        <img
          src={`${base}${item.imagePath}`}
          alt={`GTSRB class ${item.classId} reference image`}
          className="w-full border border-gray-300 mb-3 bg-gray-100"
        />

        <p className="text-xs text-gray-500 italic mb-3">
          Note: this is a GTSRB Meta reference image. The fields below show a planned prediction UI format;
          no model inference was run for this static page.
        </p>

        <table className="tbl mb-3">
          <thead><tr><th>Field</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>True label</td><td>{item.trueLabel}</td></tr>
            <tr><td>Predicted</td><td>{item.predictedLabel}</td></tr>
            <tr><td>Confidence</td><td className="tabular-nums text-gray-500">{(item.confidence * 100).toFixed(1)}%</td></tr>
          </tbody>
        </table>

        <p className="text-xs font-bold mb-1 text-gray-600">top-5 predictions:</p>
        <ol className="text-xs list-decimal list-inside text-gray-500 space-y-0.5">
          {item.topFive.map((t, i) => (
            <li key={i}>
              Class {t.classId}: {getGtsrbClassLabel(t.classId)} ({(t.prob * 100).toFixed(1)}%)
            </li>
          ))}
        </ol>
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
    <section id="gallery">
      <hr className="sec" />
      <h2 className="text-lg font-bold mb-4">4. Example Gallery</h2>

      <p className="text-sm leading-relaxed mb-2">
        The grid below shows the GTSRB class reference image for each of the 43 classes (from the
        dataset's <code>Meta/</code> directory). Click any image to view predicted class, confidence,
        and top-5 predictions. <em>All predictions shown are placeholder values</em> — the trained
        models were not deployed for this static page.
      </p>
      <p className="text-xs text-gray-500 italic mb-4">
        Figure 4. GTSRB class reference images (classes 0–42). Click to inspect.
      </p>

      <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 gap-1.5 mb-3">
        {displayed.map(item => (
          <button
            key={item.id}
            onClick={() => setSelected(item)}
            className="group relative aspect-square border border-gray-300 hover:border-gray-600 bg-gray-100 overflow-hidden"
            title={`Class ${item.classId}: ${item.trueLabel}`}
          >
            <img
              src={`${base}${item.imagePath}`}
              alt={`GTSRB class ${item.classId}: ${item.trueLabel}`}
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-white/70 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
              <span className="text-xs font-bold text-black">{item.classId}</span>
            </div>
          </button>
        ))}
      </div>

      {!showAll && (
        <p className="text-sm mb-4">
          Showing 12 of 43.{' '}
          <button onClick={() => setShowAll(true)} className="text-blue-700 underline cursor-pointer">
            Show all 43 classes.
          </button>
        </p>
      )}

      {selected && <Modal item={selected} onClose={() => setSelected(null)} />}
    </section>
  )
}
