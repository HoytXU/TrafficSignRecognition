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
      <pre className="text-xs bg-gray-50 border border-gray-300 p-3 overflow-x-auto leading-relaxed">
{`# Base dependencies
pip install -r requirements.txt
jupyter notebook expert/concepts.ipynb     # classical CV visualisations
jupyter notebook expert/task.ipynb         # classical feature grid search

# Deep learning
pip install torch torchvision timm wandb tqdm Pillow
python bonus/training/train.py --model resnet18 --epoch 10
python bonus/scripts/train_all_models.py --epoch 5
python bonus/analysis/compare_all_models.py`}
      </pre>
    </section>
  )
}
