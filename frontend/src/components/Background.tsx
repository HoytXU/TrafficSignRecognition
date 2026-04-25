const CHALLENGES = [
  { icon: '☀', label: 'Illumination', desc: 'Direct sunlight, night-time, and shade can wash out or darken sign colours.' },
  { icon: '◌', label: 'Blur & Motion', desc: 'Camera shake, driving speed, and low shutter cause motion or defocus blur.' },
  { icon: '▣', label: 'Occlusion', desc: 'Trees, vehicles, or graffiti partially block the sign — the model must infer from fragments.' },
  { icon: '⬡', label: 'Viewpoint', desc: 'Oblique angles compress circular signs to ellipses and skew text unpredictably.' },
  { icon: '⊕', label: 'Similar Classes', desc: 'Speed-limit signs differ only in a single digit inside a near-identical circular border.' },
]

export default function Background() {
  return (
    <section id="background" className="py-20 px-4 border-t border-slate-800">
      <div className="max-w-5xl mx-auto">
        <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">Why Traffic Sign Recognition?</h2>
        <p className="text-slate-500 text-sm mb-10">Motivation and key challenges</p>

        <div className="grid md:grid-cols-2 gap-10">
          {/* Why it matters */}
          <div>
            <h3 className="text-base font-semibold text-slate-200 mb-3">Real-world impact</h3>
            <p className="text-slate-400 text-sm leading-relaxed mb-3">
              Reliable traffic sign recognition is a critical perception module in Advanced Driver-Assistance Systems (ADAS) and autonomous vehicles. Misreading a stop sign or a speed limit can have immediate safety consequences.
            </p>
            <p className="text-slate-400 text-sm leading-relaxed mb-3">
              Beyond safety, accurate recognition enables automated HD-map construction, road infrastructure auditing, and compliance monitoring — all at scale.
            </p>
            <p className="text-slate-400 text-sm leading-relaxed">
              This project traces the evolution from classical hand-crafted pipelines to modern learned representations, making the performance gains of deep learning concrete and measurable on a well-known benchmark (GTSRB).
            </p>
          </div>

          {/* Key challenges */}
          <div>
            <h3 className="text-base font-semibold text-slate-200 mb-3">Key challenges</h3>
            <div className="flex flex-col gap-3">
              {CHALLENGES.map(ch => (
                <div key={ch.label} className="flex gap-3 items-start">
                  <span className="text-lg mt-0.5 w-6 text-center shrink-0">{ch.icon}</span>
                  <div>
                    <span className="text-sm font-medium text-slate-200">{ch.label} — </span>
                    <span className="text-sm text-slate-400">{ch.desc}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Dataset summary */}
        <div className="mt-10 grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { value: '58', label: 'Classes', sub: 'Dataset 1' },
            { value: '5,998', label: 'Images', sub: 'Dataset 1' },
            { value: '43', label: 'Classes', sub: 'GTSRB' },
            { value: '51,882', label: 'Images', sub: 'GTSRB' },
          ].map(stat => (
            <div key={stat.label + stat.sub} className="bg-slate-800 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-white">{stat.value}</div>
              <div className="text-xs text-slate-300">{stat.label}</div>
              <div className="text-xs text-slate-500">{stat.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
