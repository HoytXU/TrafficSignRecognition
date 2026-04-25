import { failureCases } from '../data/projectData'

export default function FailureCases() {
  const base = import.meta.env.BASE_URL

  return (
    <section id="failures">
      <hr className="sec" />
      <h2 className="text-lg font-bold mb-4">5. Failure Analysis</h2>

      <p className="text-sm leading-relaxed mb-2">
        Even high-accuracy models fail on a predictable set of cases. The examples below illustrate
        the most common failure modes, constructed from class reference images. Misclassification
        labels and confidence scores are illustrative placeholders.
      </p>
      <p className="text-xs text-gray-500 italic mb-4">
        Figure 5. Representative failure modes. All predictions are placeholder values.
      </p>

      <div className="grid sm:grid-cols-2 gap-5 mb-4">
        {failureCases.map((fc, i) => (
          <div key={i} className="flex gap-4 border border-gray-300 p-3">
            <img
              src={`${base}${fc.imagePath}`}
              alt="Input image"
              className="w-16 h-16 object-cover border border-gray-300 bg-gray-100 shrink-0"
            />
            <div className="text-sm">
              <p className="mb-1">
                <span className="text-gray-500">Predicted: </span>
                <strong className="text-red-700">{fc.predictedLabel}</strong>
                <span className="text-gray-400 text-xs ml-1">({(fc.confidence * 100).toFixed(0)}%)</span>
              </p>
              <p className="mb-1">
                <span className="text-gray-500">True label: </span>
                <strong>{fc.trueLabel}</strong>
              </p>
              <p className="text-xs text-gray-600 leading-relaxed">{fc.reason}</p>
            </div>
          </div>
        ))}
      </div>

      <p className="text-xs text-gray-500 italic">
        To generate real failure cases: run inference on the GTSRB test set with a trained checkpoint
        (<code>bonus/checkpoints/</code>), collect wrong predictions, and replace the placeholder data
        in <code>src/data/projectData.ts</code>.
      </p>
    </section>
  )
}
