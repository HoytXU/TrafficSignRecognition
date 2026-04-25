const CHALLENGES = [
  { term: 'Illumination',          def: 'Direct sunlight, night-time, and shade wash out or darken sign colours.' },
  { term: 'Blur & motion',         def: 'Camera shake and driving speed cause motion or defocus blur.' },
  { term: 'Occlusion',             def: 'Trees, vehicles, or graffiti partially block signs; the model must infer from fragments.' },
  { term: 'Viewpoint change',      def: 'Oblique angles compress circular signs to ellipses and skew text unpredictably.' },
  { term: 'Visually similar classes', def: 'Speed-limit signs share an identical circular border; only the digit distinguishes them.' },
]

export default function Background() {
  return (
    <section id="background">
      <hr className="sec" />
      <h2 className="text-lg font-bold mb-4">1. Background</h2>

      <p className="text-sm leading-relaxed mb-4">
        Reliable traffic sign recognition is a critical perception module in Advanced Driver-Assistance
        Systems (ADAS) and autonomous vehicles. A misread stop sign or speed limit can have immediate
        safety consequences. Beyond safety, accurate recognition enables automated HD-map construction,
        road infrastructure auditing, and compliance monitoring at scale.
      </p>

      <p className="text-sm leading-relaxed mb-4">
        This project makes the performance gains of deep learning concrete and measurable by tracing the
        evolution from classical hand-crafted pipelines to modern learned representations on a controlled
        benchmark.
      </p>

      <h3 className="text-sm font-bold mb-2">Key visual challenges</h3>
      <dl className="text-sm leading-relaxed mb-4">
        {CHALLENGES.map(c => (
          <div key={c.term} className="flex gap-2 mb-1">
            <dt className="font-bold shrink-0 w-44">{c.term}.</dt>
            <dd className="text-gray-700">{c.def}</dd>
          </div>
        ))}
      </dl>

      <h3 className="text-sm font-bold mb-2">Datasets</h3>
      <table className="tbl">
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Used by</th>
            <th>Images</th>
            <th>Classes</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Dataset 1</td>
            <td>Level 1, 2</td>
            <td className="tabular-nums">5,998</td>
            <td className="tabular-nums">58</td>
            <td className="text-gray-600">Custom collection; augmented variants</td>
          </tr>
          <tr>
            <td>GTSRB (Dataset 2)</td>
            <td>Level 3</td>
            <td className="tabular-nums">51,882</td>
            <td className="tabular-nums">43</td>
            <td className="text-gray-600">Pre-split train/test; ROI coordinates provided</td>
          </tr>
        </tbody>
      </table>
    </section>
  )
}
