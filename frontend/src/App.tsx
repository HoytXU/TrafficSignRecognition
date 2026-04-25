import Hero from './components/Hero'
import Background from './components/Background'
import MethodPipeline from './components/MethodPipeline'
import ModelComparison from './components/ModelComparison'
import ExampleGallery from './components/ExampleGallery'
import FailureCases from './components/FailureCases'
import MethodNotes from './components/MethodNotes'
import Links from './components/Links'

const NAV_ITEMS = [
  { id: 'hero',       label: 'Overview'   },
  { id: 'background', label: 'Background' },
  { id: 'methods',    label: 'Methods'    },
  { id: 'comparison', label: 'Results'    },
  { id: 'gallery',    label: 'Gallery'    },
  { id: 'failures',   label: 'Failures'   },
  { id: 'notes',      label: 'Notes'      },
  { id: 'links',      label: 'References' },
]

function Nav() {
  return (
    <nav className="sticky top-0 bg-white border-b border-gray-300 z-50 py-2">
      <div className="max-w-3xl mx-auto px-6 flex flex-wrap gap-x-1 gap-y-1 text-sm overflow-x-auto">
        {NAV_ITEMS.map((item, i) => (
          <span key={item.id} className="flex items-center gap-1">
            {i > 0 && <span className="text-gray-400 select-none">·</span>}
            <a href={`#${item.id}`} className="no-underline hover:underline text-blue-800">
              {item.label}
            </a>
          </span>
        ))}
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <div className="bg-white text-gray-900 min-h-screen">
      <Nav />
      <main className="max-w-3xl mx-auto px-6">
        <Hero />
        <Background />
        <MethodPipeline />
        <ModelComparison />
        <ExampleGallery />
        <FailureCases />
        <MethodNotes />
        <Links />
      </main>
      <footer className="max-w-3xl mx-auto px-6 py-8 text-xs text-gray-500 border-t border-gray-300 mt-4">
        INFH 5000 · Traffic Sign Recognition Project · Static portfolio — no live inference
      </footer>
    </div>
  )
}
