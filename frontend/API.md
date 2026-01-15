# Frontend API Contract

This document defines how the frontend consumes the backend API.

## Configuration

```typescript
// Environment variable
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";
```

---

## API Client

All API calls are in `src/lib/overlay-api.ts`.

### Upload Files

```typescript
import { uploadFiles } from "@/lib/overlay-api";

const response = await uploadFiles(fileA, fileB);
// Returns: { session_id, file_a, file_b, status, created_at, expires_at }
```

### Generate Preview

```typescript
import { generatePreview } from "@/lib/overlay-api";

const preview = await generatePreview(sessionId, "A"); // or "B"
// Returns: { session_id, which, image_id, image_url, width_px, height_px, ... }
```

### Compose Layers (Main Endpoint)

```typescript
import { composeLayers } from "@/lib/overlay-api";

const result = await composeLayers(sessionId, { auto: true });
// Returns: { session_id, status, confidence, layers[], state, bounds, warnings }
```

The response contains:
- `layers[]` - Array of layer images as base64 PNG
- `state` - Full composition state for rendering
- `bounds` - Content and geometry bounds for canvas sizing

### Generate Differences

```typescript
import { generateDifferences, DifferencesTransform } from "@/lib/overlay-api";

// Optional: pass current transform from UI if user adjusted it
const transform: DifferencesTransform = {
  scale: 1.0,
  rotation_deg: 0.0,
  translate_x: 10.5,
  translate_y: -5.3,
};

const result = await generateDifferences(sessionId, transform);
// Returns: { summary, regions[], total_regions, diff_percentage, ai_available, ... }
```

---

## Types

All types are defined in `src/lib/overlay-types.ts`.

### Key Types

```typescript
interface LayerInfo {
  id: string;
  name: string;
  source: "pair1" | "pair2";
  type: "base" | "linework" | "text" | "annotation";
  png_base64?: string;
  width: number;
  height: number;
  default_opacity: number;
}

interface LayerState {
  id: string;
  visible: boolean;
  opacity: number;
  blend_mode: BlendMode;
  color: string | null;
  transform: Transform2D;
  locked: boolean;
}

interface Transform2D {
  translate_x: number;
  translate_y: number;
  scale_x: number;
  scale_y: number;
  rotation_deg: number;
  pivot_x: number;
  pivot_y: number;
}

interface ComposeResponse {
  session_id: string;
  status: "success" | "auto_failed_fallback_manual";
  confidence: number;
  layers: LayerInfo[];
  state: CompositionState;
  bounds?: LayerBounds;
  warnings: string[];
}
```

---

## Workflow

1. **Upload** - User uploads two files via `uploadFiles()`
2. **Preview** - Generate previews with `generatePreview()` (auto-called by compose)
3. **Compose** - Call `composeLayers()` to get layers + alignment state
4. **Render** - Frontend renders layers on canvas using state
5. **Adjust** - User can drag/transform layers in the UI
6. **Analyze** - Optional: call `generateDifferences()` for AI summary
7. **Export** - Frontend exports composed image locally

---

## Error Handling

```typescript
import { OverlayApiError } from "@/lib/overlay-api";

try {
  const result = await composeLayers(sessionId);
} catch (err) {
  if (err instanceof OverlayApiError) {
    console.error(err.code, err.message, err.details);
  }
}
```

---

## Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on http://localhost:3000
