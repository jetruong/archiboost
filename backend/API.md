# Backend API Contract

This document defines the REST API provided by the backend service.

## Base URL

```
/api/v1
```

## Interactive Documentation

- Swagger UI: `GET /api/v1/docs`
- ReDoc: `GET /api/v1/redoc`
- OpenAPI JSON: `GET /api/v1/openapi.json`

---

## Endpoints

### Upload Files

```http
POST /api/v1/upload
Content-Type: multipart/form-data
```

| Field | Type | Description |
|-------|------|-------------|
| `file_a` | File | First file (PDF or PNG) |
| `file_b` | File | Second file (PDF or PNG) |

**Response** `200 OK`:
```json
{
  "session_id": "uuid",
  "file_a": { "id": "uuid", "filename": "...", "size_bytes": 12345, "uploaded_at": "..." },
  "file_b": { "id": "uuid", "filename": "...", "size_bytes": 12345, "uploaded_at": "..." },
  "status": "uploaded",
  "created_at": "...",
  "expires_at": "..."
}
```

---

### Generate Preview

```http
GET /api/v1/sessions/{session_id}/preview?which=A|B&dpi=250
```

**Response** `200 OK`:
```json
{
  "session_id": "uuid",
  "which": "A",
  "image_id": "preview_a_{id}",
  "image_url": "/api/v1/images/preview_a_{id}",
  "width_px": 1200,
  "height_px": 900,
  "dpi": 250,
  "crop_metadata": { ... },
  "processing_time_ms": 850
}
```

---

### Get Session

```http
GET /api/v1/sessions/{session_id}
```

**Response** `200 OK`:
```json
{
  "session_id": "uuid",
  "status": "uploaded|preview_ready|aligned|complete",
  "created_at": "...",
  "expires_at": "...",
  "ttl_seconds": 3600,
  "file_a": { ... },
  "file_b": { ... },
  "preview_a": { ... },
  "preview_b": { ... }
}
```

---

### Compose Layers (Primary Overlay API)

```http
POST /api/v1/sessions/{session_id}/compose
Content-Type: application/json
```

**Request**:
```json
{
  "auto": true,
  "rotation_constraint": "RIGHT_ANGLES"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto` | boolean | `true` | Use automatic alignment |
| `rotation_constraint` | string | `"RIGHT_ANGLES"` | `"NONE"`, `"RIGHT_ANGLES"`, `"SMALL_ANGLE"` |
| `manual_transform` | object | null | Manual transform override |

**Response** `200 OK`:
```json
{
  "session_id": "uuid",
  "status": "success|auto_failed_fallback_manual",
  "confidence": 0.85,
  "layers": [
    {
      "id": "layer_id",
      "name": "Drawing A",
      "source": "pair1|pair2",
      "type": "base",
      "png_base64": "data:image/png;base64,...",
      "width": 1200,
      "height": 900,
      "default_opacity": 0.7
    }
  ],
  "state": {
    "version": "1.0",
    "canvas": { "width": 1200, "height": 920, "dpi": 250, ... },
    "layers": [ { "id": "...", "transform": {...}, "opacity": 0.7, ... } ],
    "alignment": { "method": "auto", "confidence": 0.85, ... }
  },
  "bounds": {
    "content_bounds": { "x": 0, "y": 0, "width": 1220, "height": 940 },
    "geometry_bounds": { "x": 20, "y": 20, "width": 1160, "height": 860 }
  },
  "warnings": [],
  "processing_time_ms": 1250
}
```

---

### Generate Differences (AI-Powered)

```http
POST /api/v1/sessions/{session_id}/differences
Content-Type: application/json
```

**Request** (optional transform from frontend):
```json
{
  "transform": {
    "scale": 1.0,
    "rotation_deg": 0.0,
    "translate_x": 10.5,
    "translate_y": -5.3
  }
}
```

**Response** `200 OK`:
```json
{
  "session_id": "uuid",
  "summary": "AI-generated description of differences...",
  "regions": [
    { "id": 1, "bbox": [x, y, w, h], "area": 2450, "centroid": [cx, cy] }
  ],
  "total_regions": 3,
  "diff_percentage": 2.3,
  "ai_available": true,
  "model_name": "gemini-2.0-flash",
  "is_vlm": true,
  "processing_time_ms": 3200
}
```

---

### Serve Images

```http
GET /api/v1/images/{image_id}
```

Returns binary PNG image data.

---

### Health Check

```http
GET /health
```

**Response**:
```json
{
  "ok": true,
  "status": "healthy",
  "version": "0.1.0"
}
```

---

## Error Format

All errors follow this structure:

```json
{
  "detail": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": { ... }
  }
}
```

Common error codes:
- `SESSION_NOT_FOUND` (404)
- `PREVIEWS_NOT_READY` (409)
- `INVALID_FILE_TYPE` (400)
- `FILE_TOO_LARGE` (400)

---

## Running the Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
