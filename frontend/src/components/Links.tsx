import { projectLinks } from '../data/projectData'

export default function Links() {
  return (
    <section id="links">
      <hr className="sec" />
      <h2 className="text-lg font-bold mb-4">References & Resources</h2>

      {projectLinks.map(group => (
        <div key={group.category} className="mb-6">
          <h3 className="text-sm font-bold mb-2">{group.category}</h3>
          <ul className="space-y-2">
            {group.items.map(link => (
              <li key={link.label} className="text-sm">
                <a href={link.url} target="_blank" rel="noopener noreferrer" className="underline">
                  {link.label}
                </a>
                <span className="text-gray-600"> — {link.desc}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}

      <hr className="sec" />
      <h3 className="text-sm font-bold mb-2">Reproduce the results</h3>
      <p className="text-sm mb-3 leading-relaxed">
        Full reproduction depends on three components, hosted across GitHub and Hugging Face:
      </p>
      <ol className="list-decimal list-inside text-sm space-y-1 mb-4">
        <li>
          <span className="font-semibold">Data</span> — hosted on Hugging Face:{' '}
          <a
            href="https://huggingface.co/datasets/IridesParadox/TrafficSignRecognition"
            target="_blank"
            rel="noopener noreferrer"
            className="underline"
          >
            IridesParadox/TrafficSignRecognition
          </a>
        </li>
        <li>
          <span className="font-semibold">Code</span> — hosted in the GitHub repository:{' '}
          <a
            href="https://github.com/HoytXU/TrafficSignRecognition"
            target="_blank"
            rel="noopener noreferrer"
            className="underline"
          >
            HoytXU/TrafficSignRecognition
          </a>
        </li>
        <li>
          <span className="font-semibold">Checkpoints</span> — pretrained models on Hugging Face:{' '}
          <a
            href="https://huggingface.co/IridesParadox/TrafficSignRecognition_checkpoints"
            target="_blank"
            rel="noopener noreferrer"
            className="underline"
          >
            IridesParadox/TrafficSignRecognition_checkpoints
          </a>
        </li>
      </ol>

      <p className="text-sm font-semibold mb-2">Workflow</p>
      <ol className="list-decimal list-inside text-sm space-y-1 mb-4">
        <li>Clone the GitHub repository.</li>
        <li>Set up the Python environment and install the project dependencies.</li>
        <li>Download or access the dataset from Hugging Face.</li>
        <li>Download the pretrained checkpoints from Hugging Face.</li>
        <li>Run the evaluation or inference scripts / notebooks.</li>
      </ol>

      <pre className="text-xs bg-gray-50 border border-gray-300 p-3 overflow-x-auto leading-relaxed">
{`# 1. Clone the repository
git clone https://github.com/HoytXU/TrafficSignRecognition.git

# 2. Set up the Python environment
pip install -r requirements.txt
pip install torch torchvision torchaudio wandb tqdm Pillow  # deep-learning extras

# 3. Dataset      — https://huggingface.co/datasets/IridesParadox/TrafficSignRecognition
# 4. Checkpoints  — https://huggingface.co/IridesParadox/TrafficSignRecognition_checkpoints

# 5. Run evaluation / inference
jupyter notebook expert/concepts.ipynb     # classical CV visualisations
jupyter notebook expert/task.ipynb         # classical feature grid search
python bonus/analysis/compare_all_models.py`}
      </pre>
    </section>
  )
}
