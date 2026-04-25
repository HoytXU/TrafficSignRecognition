import { projectLinks } from '../data/projectData'

const CATEGORY_ICONS: Record<string, string> = {
  'Notebooks': '📓',
  'Code & Data': '💾',
  'Slides': '📊',
}

export default function Links() {
  return (
    <section id="links" className="py-20 px-4 border-t border-slate-800">
      <div className="max-w-5xl mx-auto">
        <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">Links & Resources</h2>
        <p className="text-slate-500 text-sm mb-10">Notebooks, source code, datasets, and checkpoints.</p>

        <div className="flex flex-col gap-8">
          {projectLinks.map(group => (
            <div key={group.category}>
              <div className="flex items-center gap-2 mb-4">
                <span className="text-lg">{CATEGORY_ICONS[group.category] ?? '🔗'}</span>
                <h3 className="text-sm font-semibold uppercase tracking-widest text-slate-400">
                  {group.category}
                </h3>
              </div>
              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {group.items.map(link => (
                  <a
                    key={link.label}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block rounded-xl border border-slate-700 bg-slate-800/40 p-4 hover:border-blue-500/50 hover:bg-slate-800 transition-colors group"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-slate-200 group-hover:text-white transition-colors">
                        {link.label}
                      </span>
                      <span className="text-slate-600 group-hover:text-blue-400 transition-colors text-sm">↗</span>
                    </div>
                    <p className="text-xs text-slate-500 leading-relaxed">{link.desc}</p>
                  </a>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-12 p-5 rounded-xl border border-slate-700 bg-slate-800/40">
          <h3 className="text-sm font-semibold text-slate-200 mb-2">Run it yourself</h3>
          <div className="space-y-1 text-xs text-slate-400 font-mono">
            <p><span className="text-slate-600"># Classical levels</span></p>
            <p>pip install -r requirements.txt</p>
            <p>jupyter notebook expert/concepts.ipynb</p>
            <p className="mt-2"><span className="text-slate-600"># Deep learning</span></p>
            <p>pip install torch torchvision timm wandb tqdm Pillow</p>
            <p>python bonus/training/train.py --model resnet18 --epoch 10</p>
          </div>
        </div>
      </div>
    </section>
  )
}
