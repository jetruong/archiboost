# Archiboost – Architectural Detail Comparison Tool

A web application for comparing architectural detail drawings by overlaying them with distinct colors to reveal differences.

## Problem

ArchiBoost aggregates architectural details from past projects into a centralized library. When users browse this library, they often encounter multiple versions of the same detail. Visually comparing these versions is tedious and error-prone.

This tool enables users to overlay two details, rendering each in a different color so differences and similarities are immediately apparent—helping users decide which version to keep or reuse.

## Features

- **Upload or Select**: Upload two PDF/PNG files or select from a persistent library
- **Auto Alignment**: Automatic geometric alignment using phase correlation
- **Layer Editing**: Toggle visibility, adjust opacity, manually reposition layers
- **Difference Analysis**: AI-powered summary of detected differences
- **Export**: Download the composed overlay as a PNG

## Quick Start

### Prerequisites

- Python 3.11+ with pip (or conda)
- Node.js 18+ with npm

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --reload --port 8000
```

API available at http://localhost:8000/api/v1/docs

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1" > .env.local

# Start development server
npm run dev
```

Application available at http://localhost:3000

## Usage

1. Open http://localhost:3000
2. Click **Create Overlay**
3. Select or upload two files (Drawing A and Drawing B)
4. View the auto-aligned overlay in the editor
5. Adjust layers as needed using the side panels
6. Click **Export PNG** to download

## Project Structure

```
├── frontend/               # Next.js 14 application
│   ├── src/app/           # Page routes
│   ├── src/components/    # Overlay editor components
│   └── src/lib/           # API clients and types
│
├── backend/               # FastAPI service
│   ├── app/routes/        # API endpoints
│   ├── app/services/      # Business logic (alignment, overlay, etc.)
│   └── app/models/        # Data schemas
│
├── README.md              # This file
└── CHANGELOG.md           # Version history
```

## API Reference

- `POST /api/v1/upload` – Upload two files, creates a session
- `GET /api/v1/sessions/{id}/preview` – Generate cropped preview
- `POST /api/v1/sessions/{id}/compose` – Get layer data and alignment state
- `POST /api/v1/sessions/{id}/differences` – Generate AI-powered diff summary

See [backend/API.md](backend/API.md) and [frontend/API.md](frontend/API.md) for detailed contracts.

## Technical Approach

**Alignment**: The system uses a geometry-first pipeline optimized for architectural CAD drawings:
1. Extract linework via edge detection with text suppression
2. Evaluate rotation candidates (0°, 90°, 180°, 270°)
3. Estimate translation via phase correlation
4. Optional feature-based refinement for scale estimation
5. Guardrails reject extreme transforms

**Rendering**: The backend provides individual layer images and state; the frontend handles all composition. This preserves full editability without server round-trips.

**Colors**: Default overlay uses red (Drawing A) and cyan (Drawing B). Where lines overlap, they appear dark/black, indicating matching geometry.

## Assumptions

- Input files are single-page architectural details (PDF or PNG)
- Drawings are axis-aligned (no significant rotation between versions)
- File size limit: 25 MB per file
- Sessions expire after 1 hour

## Limitations

- No user authentication
- No multi-page PDF support
- Auto-alignment may fail on very different drawings (manual fallback available)
- No perspective correction for scanned/photographed drawings
