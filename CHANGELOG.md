# Changelog

## v1.0.0 – Initial Submission

### Overview

Archiboost is a tool for comparing architectural detail drawings. Users upload two PDF or PNG files, and the system automatically aligns and overlays them, rendering each drawing in a distinct color to highlight differences and similarities. This helps architects quickly evaluate which version of a detail to keep or reuse.

### Frontend

- **Upload Flow**: Select two files from library or upload new files directly
- **Overlay Editor**: Layer-based canvas with interactive controls
  - Toggle individual layer visibility
  - Adjust layer opacity
  - Drag to manually reposition layers
  - Keyboard shortcuts for nudge (arrow keys) and undo/redo (Cmd+Z)
- **Canvas Rendering**: White background with no visual cutoff; full drawing content visible
- **Smart Analysis**: AI-powered difference detection with region highlighting
- **Export**: PNG export of the composed overlay at full resolution
- **Library Management**: Browse, upload, delete, and reuse files across sessions

### Backend

- **Alignment Pipeline**:
  - Automatic alignment using phase correlation and feature matching
  - Geometry-first approach optimized for architectural CAD drawings
  - Text suppression to avoid spurious matches on labels
  - Fallback to manual anchor point selection if auto-alignment fails
- **Overlay Generation**:
  - Layer-based composition (not flattened server-side)
  - Configurable tint colors (default: red/cyan)
  - Line extraction via grayscale thresholding
- **PDF/PNG Handling**:
  - PDF rasterization at 250 DPI using PyMuPDF
  - Automatic whitespace cropping with configurable padding
  - Support for both PDF and PNG input formats
- **API Design**:
  - RESTful endpoints with session-based workflow
  - Stateless design with temporary session storage
  - OpenAPI documentation at `/api/v1/docs`
  - Optional AI-powered difference summarization via Gemini

### Architecture Decisions

- **Frontend (Next.js 14 + TypeScript)**: Chosen for fast iteration, type safety, and server components. Tailwind CSS for styling with a custom dark theme.
- **Backend (FastAPI + Python)**: Enables rapid API development with automatic validation. Python ecosystem provides mature image processing libraries (OpenCV, PIL, PyMuPDF).
- **Layer-Based Composition**: Frontend handles all rendering from individual layer images. Backend never produces a flattened composite, preserving full editability.
- **Session Model**: Temporary sessions expire after 1 hour. No user authentication required for v1.
- **Canvas Coordinate System**: All transforms operate in pixel coordinates relative to cropped preview images, simplifying frontend-backend communication.

### Out of Scope for v1

- User authentication and persistent accounts
- Multi-page PDF support
- Automatic anchor point detection from user clicks
- Perspective transform correction (for scanned drawings)
- Real-time collaborative editing
- Batch comparison of multiple drawing pairs
- Version history within sessions
