// ─── Types ───────────────────────────────────────────────────────────────────

export interface DLModel {
  name: string;
  shortName: string;
  type: 'classic-dl' | 'transfer' | 'transformer' | 'custom';
  testAcc: number;
  f1: number;
  params: string;
  speed: 'very fast' | 'fast' | 'medium' | 'slow';
  suitability: string;
  bestEpoch: number;
  trainAcc: number;
  trainLoss: number;
  note: string;
  lr: string;
}

export interface ClassicalMethod {
  name: string;
  dataset: string;
  testAcc: string;
  accIsReal: boolean;
  note: string;
}

export interface PipelineStep {
  label: string;
  detail: string;
}

export interface PipelineMethod {
  id: string;
  name: string;
  color: 'amber' | 'emerald' | 'blue' | 'purple';
  steps: PipelineStep[];
  highlights: string[];
}

export interface GalleryItem {
  id: number;
  imagePath: string;
  classId: number;
  trueLabel: string;
  // All predictions are PLACEHOLDER — no inference was run
  predictedLabel: string;
  confidence: number;
  topFive: { classId: number; prob: number }[];
  isPlaceholder: true;
}

export interface FailureCase {
  imagePath: string;
  trueLabel: string;
  predictedLabel: string;
  confidence: number;
  reason: string;
  isPlaceholder: true;
}

export interface MethodNote {
  id: string;
  title: string;
  color: 'amber' | 'blue' | 'purple';
  bullets: string[];
}

export interface LinkItem {
  label: string;
  url: string;
  desc: string;
}

export interface LinkGroup {
  category: string;
  items: LinkItem[];
}

// ─── Deep Learning Models (REAL metrics from bonus/analysis/model_comparison_summary.json) ──

export const dlModels: DLModel[] = [
  {
    name: 'ViT-B/16',
    shortName: 'ViT',
    type: 'transformer',
    testAcc: 0.9804,
    f1: 0.9696,
    params: '86M',
    speed: 'slow',
    suitability: 'Server / GPU',
    bestEpoch: 4,
    trainAcc: 0.9916,
    trainLoss: 0.04,
    note: 'Vision Transformer with global self-attention over 16×16 patches. Highest accuracy.',
    lr: '0.0001',
  },
  {
    name: 'ResNet18',
    shortName: 'ResNet18',
    type: 'transfer',
    testAcc: 0.9717,
    f1: 0.9477,
    params: '11M',
    speed: 'fast',
    suitability: 'Server / Cloud',
    bestEpoch: 3,
    trainAcc: 0.9946,
    trainLoss: 0.0325,
    note: 'Residual connections prevent vanishing gradients. Reliable generalization.',
    lr: '0.001',
  },
  {
    name: 'AlexNet',
    shortName: 'AlexNet',
    type: 'transfer',
    testAcc: 0.9687,
    f1: 0.9545,
    params: '61M',
    speed: 'fast',
    suitability: 'Server / Cloud',
    bestEpoch: 4,
    trainAcc: 0.9848,
    trainLoss: 0.0551,
    note: 'Best efficiency-accuracy balance in this study. Large params but fast inference.',
    lr: '0.001',
  },
  {
    name: 'VGG16',
    shortName: 'VGG16',
    type: 'transfer',
    testAcc: 0.9616,
    f1: 0.9413,
    params: '138M',
    speed: 'slow',
    suitability: 'Server (high mem)',
    bestEpoch: 4,
    trainAcc: 0.9799,
    trainLoss: 0.0751,
    note: 'Deep sequential architecture with strong feature hierarchies. High memory cost.',
    lr: '0.001',
  },
  {
    name: 'SqueezeNet',
    shortName: 'SqzNet',
    type: 'transfer',
    testAcc: 0.9601,
    f1: 0.9462,
    params: '1.2M',
    speed: 'very fast',
    suitability: 'Edge / Embedded',
    bestEpoch: 5,
    trainAcc: 0.9794,
    trainLoss: 0.0705,
    note: 'Fire modules compress channels aggressively. Best for edge deployment.',
    lr: '0.001',
  },
  {
    name: 'MY_NET',
    shortName: 'MY_NET',
    type: 'custom',
    testAcc: 0.9386,
    f1: 0.9267,
    params: '~2M',
    speed: 'fast',
    suitability: 'Mobile / Embedded',
    bestEpoch: 4,
    trainAcc: 0.9867,
    trainLoss: 0.0702,
    note: 'Custom CNN with residual blocks and batch norm, designed for traffic signs.',
    lr: '0.001',
  },
  {
    name: 'LeNet',
    shortName: 'LeNet',
    type: 'classic-dl',
    testAcc: 0.8633,
    f1: 0.8151,
    params: '60K',
    speed: 'very fast',
    suitability: 'Baseline only',
    bestEpoch: 3,
    trainAcc: 0.9533,
    trainLoss: 0.1589,
    note: 'Classic 1998 CNN trained from scratch. Limited capacity; serves as baseline.',
    lr: '0.001',
  },
];

// ─── Classical Methods (PLACEHOLDER — exact results not saved) ────────────────

export const classicalMethods: ClassicalMethod[] = [
  {
    name: 'HOG + SVM',
    dataset: 'Dataset 1 (58 classes, 5,998 images)',
    testAcc: '~92%',
    accIsReal: false,
    note: 'HOG feature extraction (9 orientations, 8×8 cells) with a linear SVM. Fast, interpretable baseline.',
  },
  {
    name: 'Best Classical Pipeline',
    dataset: 'Dataset 1 (58 classes, 5,998 images)',
    testAcc: 'varies by config',
    accIsReal: false,
    note: 'Grid of preprocessing × feature × classifier combinations. Run expert/task.ipynb for exact results.',
  },
];

// ─── Method Pipeline Explorer ─────────────────────────────────────────────────

export const pipelineMethods: PipelineMethod[] = [
  {
    id: 'hog-svm',
    name: 'HOG + SVM',
    color: 'amber',
    steps: [
      { label: 'Input', detail: 'Variable-size image' },
      { label: 'Preprocess', detail: 'Resize 64×64, Grayscale' },
      { label: 'HOG', detail: '9 orientations · 8×8 cells → 128-d vector' },
      { label: 'Linear SVM', detail: 'Maximum-margin hyperplane' },
      { label: 'Prediction', detail: '58-class label' },
    ],
    highlights: [
      'No GPU required',
      'Fast CPU inference',
      'Interpretable gradients',
      '~92% accuracy (approx., not saved)',
    ],
  },
  {
    id: 'classical',
    name: 'Classical Feature Engineering',
    color: 'emerald',
    steps: [
      { label: 'Input', detail: 'Variable-size image' },
      { label: 'Preprocessing', detail: 'Simple / Blur / HistEq / Advanced' },
      { label: 'Feature Extraction', detail: 'HOG · LBP · Pyramid · FFT · Hu Moments · Multi' },
      { label: 'Classifier', detail: 'SVM · RF · kNN · DT · Naive Bayes · MLP' },
      { label: 'Prediction', detail: '58-class label' },
    ],
    highlights: [
      '4 preprocessing variants',
      '6 feature extractors (7-d to 3,024-d)',
      '6 classifiers compared',
      'See expert/task.ipynb for results',
    ],
  },
  {
    id: 'cnn',
    name: 'CNN / Transfer Learning',
    color: 'blue',
    steps: [
      { label: 'Input', detail: '224×224 px RGB' },
      { label: 'Normalize', detail: 'ImageNet mean/std' },
      { label: 'CNN Backbone', detail: 'ResNet18 · VGG16 · AlexNet · SqueezeNet (pretrained)' },
      { label: 'FC Head', detail: 'Fine-tuned linear → 43 classes' },
      { label: 'Prediction', detail: 'Softmax over 43 GTSRB classes' },
    ],
    highlights: [
      'ImageNet pretrained weights',
      'Transfer learning ≈ +10–15% vs scratch',
      'GTSRB: 51,882 images · 43 classes',
      'Adam · LR 0.001 · 5 epochs',
    ],
  },
  {
    id: 'vit',
    name: 'Vision Transformer (ViT-B/16)',
    color: 'purple',
    steps: [
      { label: 'Input', detail: '224×224 px RGB' },
      { label: 'Patch Embed', detail: '16×16 patches → 196 tokens + [CLS]' },
      { label: 'Transformer Encoder', detail: '12 blocks · 12-head attention · 768-d embeddings' },
      { label: '[CLS] Token', detail: 'Aggregates global context via self-attention' },
      { label: 'Prediction', detail: 'MLP head → 43-class label  (98.04% test acc)' },
    ],
    highlights: [
      'Global attention — no local bias',
      'Best accuracy: 98.04%',
      'Pretrained on ImageNet-21K',
      'LR 0.0001 (lower than CNNs)',
    ],
  },
];

// ─── Example Gallery (PLACEHOLDER predictions — class reference images only) ──
// Images must be copied to public/images/meta/ before build.
// Run: npm run copy-assets

const _conf = [
  0.94, 0.91, 0.97, 0.88, 0.95, 0.92, 0.89, 0.96, 0.93, 0.87,
  0.98, 0.90, 0.94, 0.91, 0.96, 0.85, 0.93, 0.97, 0.88, 0.92,
  0.95, 0.89, 0.94, 0.91, 0.97, 0.86, 0.93, 0.96, 0.90, 0.94,
  0.92, 0.88, 0.97, 0.91, 0.95, 0.89, 0.93, 0.96, 0.87, 0.94,
  0.91, 0.98, 0.92,
];

export const galleryItems: GalleryItem[] = Array.from({ length: 43 }, (_, i) => {
  const c = _conf[i];
  const rem = +(1 - c).toFixed(3);
  return {
    id: i,
    imagePath: `images/meta/${i}.png`,
    classId: i,
    trueLabel: `Class ${i}`,
    predictedLabel: `Class ${i}`,
    confidence: c,
    topFive: [
      { classId: i,            prob: c },
      { classId: (i + 5)  % 43, prob: +(rem * 0.50).toFixed(3) },
      { classId: (i + 10) % 43, prob: +(rem * 0.25).toFixed(3) },
      { classId: (i + 15) % 43, prob: +(rem * 0.15).toFixed(3) },
      { classId: (i + 20) % 43, prob: +(rem * 0.07).toFixed(3) },
    ],
    isPlaceholder: true,
  };
});

// ─── Failure Cases (PLACEHOLDER — illustrate known failure modes) ─────────────

export const failureCases: FailureCase[] = [
  {
    imagePath: 'images/meta/1.png',
    trueLabel: 'Class 1',
    predictedLabel: 'Class 2',
    confidence: 0.51,
    reason: 'Adjacent speed-limit signs share identical circular shape; only the digit differs. Blurry or low-res input makes the digit ambiguous.',
    isPlaceholder: true,
  },
  {
    imagePath: 'images/meta/11.png',
    trueLabel: 'Class 11',
    predictedLabel: 'Class 30',
    confidence: 0.43,
    reason: 'Triangular warning signs share the same red-border structure. Occlusion of the inner pictogram causes the model to fall back on shape alone.',
    isPlaceholder: true,
  },
  {
    imagePath: 'images/meta/27.png',
    trueLabel: 'Class 27',
    predictedLabel: 'Class 18',
    confidence: 0.48,
    reason: 'Pedestrian and general-warning triangular signs look nearly identical at a glance. Viewpoint change distorts the inner figure.',
    isPlaceholder: true,
  },
  {
    imagePath: 'images/meta/40.png',
    trueLabel: 'Class 40',
    predictedLabel: 'Class 12',
    confidence: 0.39,
    reason: 'Class 40 (roundabout) is one of the rarest in GTSRB. Under-representation leads to poor recall; the circular shape matches class 12.',
    isPlaceholder: true,
  },
  {
    imagePath: 'images/meta/37.png',
    trueLabel: 'Class 37',
    predictedLabel: 'Class 36',
    confidence: 0.55,
    reason: 'Go-straight-or-right (class 36) and go-straight-or-left (class 37) are mirror images. Extreme lighting flattens the directional arrow.',
    isPlaceholder: true,
  },
];

// ─── Method Notes (expandable) ────────────────────────────────────────────────

export const methodNotes: MethodNote[] = [
  {
    id: 'hog',
    title: 'HOG — Histogram of Oriented Gradients',
    color: 'amber',
    bullets: [
      'Computes image gradients and bins their orientations into a histogram per 8×8-pixel cell.',
      'Cells are grouped into overlapping blocks and contrast-normalised — robust to local illumination changes.',
      'Result: a fixed-length descriptor (128-d for the settings used here) capturing shape and texture.',
      'Strength: fast, no learning needed, interpretable. Limitation: no global context, fixed spatial resolution.',
    ],
  },
  {
    id: 'svm',
    title: 'SVM — Support Vector Machine',
    color: 'amber',
    bullets: [
      'Finds the maximum-margin hyperplane that separates classes in feature space.',
      'The RBF kernel implicitly maps features to higher dimensions, handling non-linear decision boundaries.',
      'Only a small fraction of training examples (support vectors) define the boundary — memory efficient.',
      'Strength: strong theoretical guarantees, works well in high dimensions. Limitation: slow to train on large datasets.',
    ],
  },
  {
    id: 'cnn',
    title: 'CNN — Convolutional Neural Networks',
    color: 'blue',
    bullets: [
      'Convolutional layers learn hierarchical local feature detectors: edges → textures → semantic shapes.',
      'Pooling reduces spatial resolution while preserving dominant activations — approximate translation invariance.',
      'Transfer learning reuses ImageNet weights, dramatically reducing the labelled data and training time needed.',
      'Strength: state-of-the-art on image tasks, end-to-end learning. Limitation: needs GPU, less interpretable.',
    ],
  },
  {
    id: 'vit',
    title: 'ViT — Vision Transformer',
    color: 'purple',
    bullets: [
      'Splits the image into 16×16 px patches, linearly embeds each, and prepends a learnable [CLS] token.',
      'Multi-head self-attention computes pairwise relationships between all patch tokens simultaneously.',
      'No local inductive bias — the model must learn spatial structure purely from data and position encodings.',
      'Strength: global context, excellent scaling with compute. Limitation: needs large pretraining; O(n²) attention.',
    ],
  },
];

// ─── Links ────────────────────────────────────────────────────────────────────

export const projectLinks: LinkGroup[] = [
  {
    category: 'Notebooks',
    items: [
      {
        label: 'Classical CV Concepts',
        url: 'https://nbviewer.org/github/HoytXU/TrafficSignRecongnition/blob/master/expert/concepts.ipynb',
        desc: 'Feature extraction visualisations: HOG, LBP, Pyramid, FFT, Hu Moments, PCA, and 6 classifiers.',
      },
      {
        label: 'Deep Learning Visualisation',
        url: 'https://nbviewer.org/github/HoytXU/TrafficSignRecongnition/blob/master/bonus/visualization.ipynb',
        desc: 'CNN feature maps (layer by layer), Grad-CAM saliency, and ViT attention maps.',
      },
    ],
  },
  {
    category: 'Code & Data',
    items: [
      {
        label: 'GitHub Repository',
        url: 'https://github.com/HoytXU/TrafficSignRecongnition',
        desc: 'Full source code, notebooks, training scripts, and documentation.',
      },
      {
        label: 'Hugging Face Dataset',
        url: 'https://huggingface.co/datasets/IridesParadox/TrafficSignRecognition',
        desc: 'Dataset 1 (58 classes, 5,998 images) and GTSRB Dataset 2 (43 classes, 51,882 images).',
      },
      {
        label: 'Pre-trained Checkpoints',
        url: 'https://huggingface.co/IridesParadox/TrafficSignRecognition_checkpoints',
        desc: 'Saved model weights for all 7 architectures (LeNet → ViT-B/16).',
      },
    ],
  },
  {
    category: 'Slides',
    items: [
      {
        label: 'Presentation Slides',
        url: 'https://github.com/HoytXU/TrafficSignRecongnition/blob/master/assets/slides/talk.pdf',
        desc: 'Project overview, methodology, and results — NUS Summer Camp final presentation.',
      },
    ],
  },
];
