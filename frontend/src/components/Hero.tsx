const METHOD_CARDS = [
  {
    color: 'amber',
    label: 'HOG + SVM',
    desc: 'Hand-crafted gradient histograms feed a maximum-margin classifier. Fast, interpretable, no GPU.',
    badge: '~92% acc (approx.)',
    badgeTitle: 'Approximate — exact result not saved',
    dataset: 'Dataset 1 · 58 classes',
  },
  {
    color: 'emerald',
    label: 'Classical Feature Engineering',
    desc: 'Grid search across preprocessing, feature extractors (HOG, LBP, Pyramid…), and 6 classifiers.',
    badge: 'varies by config',
    badgeTitle: 'Run expert/task.ipynb for exact numbers',
    dataset: 'Dataset 1 · 58 classes',
  },
  {
    color: 'blue',
    label: 'Deep Learning & Transfer',
    desc: '7 architectures from LeNet to ViT-B/16. Transfer learning from ImageNet on GTSRB.',
    badge: '86–98% acc (real)',
    badgeTitle: 'Real metrics from model_comparison_summary.json',
    dataset: 'GTSRB · 43 classes',
  },
]

const colorCls = {
  amber:   { border: 'border-amber-500/40',   bg: 'bg-amber-500/5',   text: 'text-amber-400',   badge: 'bg-amber-500/10 text-amber-300 border border-amber-500/20' },
  emerald: { border: 'border-emerald-500/40', bg: 'bg-emerald-500/5', text: 'text-emerald-400', badge: 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20' },
  blue:    { border: 'border-blue-500/40',    bg: 'bg-blue-500/5',    text: 'text-blue-400',    badge: 'bg-blue-500/10 text-blue-300 border border-blue-500/20' },
  purple:  { border: 'border-purple-500/40',  bg: 'bg-purple-500/5',  text: 'text-purple-400',  badge: 'bg-purple-500/10 text-purple-300 border border-purple-500/20' },
} as const

export default function Hero() {
  return (
    <section id="hero" className="min-h-screen flex items-center justify-center pt-16 pb-16 px-4">
      <div className="max-w-5xl mx-auto text-center">
        <div className="inline-block px-3 py-1 rounded-full text-xs font-medium bg-slate-700 text-slate-300 mb-6">
          NUS Summer Camp · Final Project
        </div>

        <h1 className="text-4xl md:text-6xl font-bold text-white mb-5 leading-tight tracking-tight">
          Traffic Sign
          <span className="block bg-gradient-to-r from-amber-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
            Recognition
          </span>
        </h1>

        <p className="text-lg md:text-xl text-slate-400 mb-3 max-w-2xl mx-auto leading-relaxed">
          From hand-crafted visual features to modern learned representations.
        </p>
        <p className="text-sm text-slate-500 mb-14">
          A progressive study: HOG + SVM → classical feature engineering → CNN transfer learning → Vision Transformer
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {METHOD_CARDS.map(card => {
            const c = colorCls[card.color as keyof typeof colorCls]
            return (
              <div
                key={card.label}
                className={`rounded-xl border ${c.border} ${c.bg} p-5 text-left`}
              >
                <div className={`text-xs font-semibold uppercase tracking-widest mb-2 ${c.text}`}>
                  {card.label}
                </div>
                <p className="text-sm text-slate-300 mb-4 leading-relaxed">{card.desc}</p>
                <div className="flex flex-col gap-2">
                  <span
                    title={card.badgeTitle}
                    className={`text-xs px-2 py-0.5 rounded-full self-start cursor-help ${c.badge}`}
                  >
                    {card.badge}
                  </span>
                  <span className="text-xs text-slate-500">{card.dataset}</span>
                </div>
              </div>
            )
          })}
        </div>

        <div className="mt-12 flex justify-center gap-3 flex-wrap">
          <a href="#methods" className="px-5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors">
            Explore Methods
          </a>
          <a href="#comparison" className="px-5 py-2 rounded-lg border border-slate-600 hover:border-slate-400 text-slate-300 hover:text-white text-sm font-medium transition-colors">
            Model Comparison
          </a>
        </div>
      </div>
    </section>
  )
}
