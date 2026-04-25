import Hero from './components/Hero'
import Background from './components/Background'
import MethodPipeline from './components/MethodPipeline'
import ModelComparison from './components/ModelComparison'
import ExampleGallery from './components/ExampleGallery'
import FailureCases from './components/FailureCases'
import MethodNotes from './components/MethodNotes'
import Links from './components/Links'

const NAV_ITEMS = [
  { id: 'hero',       label: 'Overview' },
  { id: 'background', label: 'Background' },
  { id: 'methods',    label: 'Methods' },
  { id: 'comparison', label: 'Comparison' },
  { id: 'gallery',    label: 'Gallery' },
  { id: 'failures',   label: 'Failures' },
  { id: 'notes',      label: 'Notes' },
  { id: 'links',      label: 'Links' },
]

function Nav() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900/90 backdrop-blur-sm border-b border-slate-700/60">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-6 overflow-x-auto">
        <span className="font-bold text-white whitespace-nowrap text-sm tracking-wide shrink-0">
          Traffic Sign Recognition
        </span>
        <div className="flex gap-1">
          {NAV_ITEMS.map(item => (
            <a
              key={item.id}
              href={`#${item.id}`}
              className="px-3 py-1 rounded text-xs text-slate-400 hover:text-white hover:bg-slate-700 transition-colors whitespace-nowrap"
            >
              {item.label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <div className="bg-slate-900 text-slate-100 min-h-screen">
      <Nav />
      <main>
        <Hero />
        <Background />
        <MethodPipeline />
        <ModelComparison />
        <ExampleGallery />
        <FailureCases />
        <MethodNotes />
        <Links />
      </main>
      <footer className="border-t border-slate-700 py-8 text-center text-slate-500 text-sm">
        NUS Summer Camp · Traffic Sign Recognition Project · Static portfolio — no live inference
      </footer>
    </div>
  )
}
