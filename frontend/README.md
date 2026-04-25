# Frontend — Traffic Sign Recognition Portfolio

Static portfolio page for the Traffic Sign Recognition project. Built with Vite + React + TypeScript + Tailwind CSS. No backend required; all data is precomputed and static.

## Running locally

```bash
cd frontend
npm install
npm run copy-assets   # copies images from datasets/ and bonus/analysis/ into public/
npm run dev
```

The dev server starts at `http://localhost:5173/TrafficSignRecognition/` (base path applied).

> **Note:** `npm run copy-assets` must be run at least once before `dev` or `build`. It copies:
> - `datasets/dataset2/Meta/0.png … 42.png` → `public/images/meta/`
> - `bonus/analysis/model_comparison.png` → `public/images/`
> - `bonus/analysis/data_analysis.png` → `public/images/`

## Building for GitHub Pages

```bash
cd frontend
npm run build   # runs copy-assets first (prebuild hook), then tsc + vite build
```

Output goes to `../docs/`. GitHub Pages is configured to serve from the `docs/` folder on the `master` branch.

### GitHub Pages setup (one-time)

1. Go to **Settings → Pages** in your GitHub repository.
2. Under **Source**, choose **Deploy from a branch**.
3. Select branch `master`, folder `/docs`.
4. Save. The site will be live at `https://HoytXU.github.io/TrafficSignRecognition/`.

### Verify the base path

`vite.config.ts` sets `base: '/TrafficSignRecognition/'`. This must exactly match your GitHub repository name (including any capitalisation). If your repo name differs, update the `base` field and the `--base` flag in the `preview` script in `package.json`.

## What is static vs what needs a backend

| Feature | Status |
|---|---|
| Hero, Background, Method Notes | Static — fully built |
| Method Pipeline Explorer (tabs) | Static — hardcoded pipeline steps |
| Model Comparison table (sort/filter) | Static — real metrics from `model_comparison_summary.json` |
| Model Comparison charts | Static — PNG images from `bonus/analysis/` |
| Class Reference Gallery images | Static — GTSRB Meta/ reference images |
| Planned prediction UI format | **Placeholder** — hardcoded in `src/data/projectData.ts` |
| Failure Cases | **Placeholder** — illustrative examples only |
| Links | Static — external URLs |

## Adding live inference (future work)

To replace placeholder predictions with real model outputs:

1. Run the trained PyTorch models (`bonus/checkpoints/*.pt`) behind a REST API (FastAPI, Flask, or TorchServe).
2. In `ExampleGallery.tsx`, replace the static `galleryItems` lookup with a `fetch()` call to the API endpoint.
3. Pass the selected image path (or upload) and receive `{ predictedClass, confidence, topFive }`.
4. Remove the `isPlaceholder` flag from the displayed data.

The frontend architecture is already structured for this: `projectData.ts` holds the static data, and each component receives it as props — swapping the data source requires changes only in the component, not the data file.

## Project structure

```
frontend/
├── public/
│   └── images/
│       ├── meta/          (43 GTSRB class PNGs — copied from datasets/dataset2/Meta/)
│       ├── model_comparison.png
│       └── data_analysis.png
├── scripts/
│   └── copy-assets.sh     (copies images before build)
├── src/
│   ├── components/
│   │   ├── Hero.tsx
│   │   ├── Background.tsx
│   │   ├── MethodPipeline.tsx
│   │   ├── ModelComparison.tsx
│   │   ├── ExampleGallery.tsx
│   │   ├── FailureCases.tsx
│   │   ├── MethodNotes.tsx
│   │   └── Links.tsx
│   ├── data/
│   │   └── projectData.ts   (all static data; mark real vs placeholder)
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── index.html
├── package.json
├── tailwind.config.js
├── postcss.config.js
├── tsconfig.json
└── vite.config.ts
```
