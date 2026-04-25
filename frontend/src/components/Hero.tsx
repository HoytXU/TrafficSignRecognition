const METHODS = [
  { level: 'Level 1', name: 'HOG + SVM',                   acc: '~92%*',         dataset: 'Dataset 1, 58 classes' },
  { level: 'Level 2', name: 'Classical Feature Engineering', acc: 'varies†',       dataset: 'Dataset 1, 58 classes' },
  { level: 'Level 3', name: 'Deep Learning / Transfer',     acc: '86.3 – 98.0%',  dataset: 'GTSRB, 43 classes'    },
]

export default function Hero() {
  return (
    <section id="hero" className="pt-12 pb-2">
      <h1 className="text-3xl font-bold leading-snug mb-1">
        Traffic Sign Recognition:<br />
        <span className="font-normal text-2xl">From Hand-Crafted Visual Features to Learned Representations</span>
      </h1>

      <p className="text-sm text-gray-600 italic mt-2 mb-1">INFH 5000 — Final Project</p>

      <hr className="sec" />

      <h2 className="text-sm font-bold uppercase tracking-widest mb-2">Abstract</h2>
      <p className="text-sm leading-relaxed text-gray-800">
        We study traffic sign recognition through three progressive levels of complexity.
        First, hand-crafted HOG features feed a linear SVM classifier on a 58-class dataset (~92% accuracy).
        Second, a systematic grid search evaluates combinations of four preprocessing methods,
        six feature extractors (HOG, LBP, Pyramid, FFT, Hu Moments, Multi-feature), and six classifiers.
        Third, we apply transfer learning with seven deep architectures — LeNet, ResNet18, VGG16, AlexNet,
        SqueezeNet, a custom residual CNN, and Vision Transformer (ViT-B/16) — on the German Traffic Sign
        Recognition Benchmark (GTSRB, 43 classes, 51,882 images). Our best model, ViT-B/16,
        achieves <strong>98.04% test accuracy</strong> after five epochs, demonstrating the advantage
        of pretrained self-attention mechanisms for fine-grained visual classification.
      </p>

      <hr className="sec" />

      <h2 className="text-sm font-bold uppercase tracking-widest mb-3">Methods Summary</h2>
      <table className="tbl mb-2">
        <thead>
          <tr>
            <th>Level</th>
            <th>Approach</th>
            <th>Test Accuracy</th>
            <th>Dataset</th>
          </tr>
        </thead>
        <tbody>
          {METHODS.map(m => (
            <tr key={m.level}>
              <td className="text-gray-500">{m.level}</td>
              <td>{m.name}</td>
              <td className="tabular-nums">{m.acc}</td>
              <td className="text-gray-600">{m.dataset}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-gray-500 mb-1">
        * Approximate — exact result not saved. Run <code>beginner/starter.py</code> to reproduce.
      </p>
      <p className="text-xs text-gray-500">
        † Varies by preprocessing × feature × classifier combination. Run <code>expert/task.ipynb</code> for exact numbers.
      </p>
    </section>
  )
}
