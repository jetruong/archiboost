/**
 * Overlay API Client
 * 
 * Client for the layer-based overlay composition API.
 */

import { z } from "zod";
import { ComposeRequest, ComposeResponse, CompositionState } from "./overlay-types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

// ============================================================
// Schemas
// ============================================================

const Transform2DSchema = z.object({
  translate_x: z.number(),
  translate_y: z.number(),
  scale_x: z.number(),
  scale_y: z.number(),
  rotation_deg: z.number(),
  pivot_x: z.number(),
  pivot_y: z.number(),
});

const LayerInfoSchema = z.object({
  id: z.string(),
  name: z.string(),
  source: z.enum(["pair1", "pair2"]),
  type: z.enum(["base", "linework", "text", "annotation"]),
  png_url: z.string().nullable().optional(),
  png_base64: z.string().nullable().optional(),
  width: z.number(),
  height: z.number(),
  default_color: z.string().nullable().optional(),
  default_opacity: z.number(),
});

const LayerStateSchema = z.object({
  id: z.string(),
  visible: z.boolean(),
  opacity: z.number(),
  blend_mode: z.enum(["normal", "multiply", "screen", "overlay", "darken", "lighten"]),
  color: z.string().nullable().optional(),
  transform: Transform2DSchema,
  locked: z.boolean(),
});

const CanvasStateSchema = z.object({
  width: z.number(),
  height: z.number(),
  dpi: z.number(),
  background_color: z.string(),
  coordinate_system: z.string(),
});

const AlignmentStateSchema = z.object({
  method: z.string(),
  confidence: z.number(),
  transform: Transform2DSchema,
  phase_response: z.number().nullable().optional(),
  overlap_ratio: z.number().nullable().optional(),
  refinement_method: z.string().nullable().optional(),
  warnings: z.array(z.string()),
});

const ColorPaletteSchema = z.object({
  name: z.string(),
  colors: z.record(z.string()),
});

const CompositionDefaultsSchema = z.object({
  palette: ColorPaletteSchema,
  snap_angles: z.array(z.number()),
  snap_enabled: z.boolean(),
  grid_size: z.number(),
});

const CompositionStateSchema = z.object({
  version: z.string(),
  created_at: z.string(),
  canvas: CanvasStateSchema,
  layers: z.array(LayerStateSchema),
  alignment: AlignmentStateSchema,
  defaults: CompositionDefaultsSchema,
});

// Bounding box schema for canvas sizing and export cropping
const BoundingBoxSchema = z.object({
  x: z.number(),
  y: z.number(),
  width: z.number(),
  height: z.number(),
});

const LayerBoundsInfoSchema = z.object({
  content_bounds: BoundingBoxSchema,
  geometry_bounds: BoundingBoxSchema,
});

const ComposeResponseSchema = z.object({
  session_id: z.string(),
  status: z.enum(["success", "auto_failed_fallback_manual"]),
  confidence: z.number(),
  layers: z.array(LayerInfoSchema),
  state: CompositionStateSchema,
  // Bounds information for canvas sizing (contentBounds) and export cropping (geometryBounds)
  bounds: LayerBoundsInfoSchema.nullable().optional(),
  warnings: z.array(z.string()),
  reason: z.string().nullable().optional(),
  processing_time_ms: z.number(),
  generated_at: z.string(),
});

// ============================================================
// API Error
// ============================================================

export class OverlayApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: unknown
  ) {
    super(message);
    this.name = "OverlayApiError";
  }
}

// ============================================================
// Session Upload Types
// ============================================================

export interface UploadResponse {
  session_id: string;
  file_a: {
    id: string;
    filename: string;
    size_bytes: number;
    uploaded_at: string;
  };
  file_b: {
    id: string;
    filename: string;
    size_bytes: number;
    uploaded_at: string;
  };
  status: string;
  created_at: string;
  expires_at: string;
}

export interface PreviewResponse {
  session_id: string;
  which: string;
  image_id: string;
  image_url: string;
  width_px: number;
  height_px: number;
  dpi: number;
  processing_time_ms: number;
}

// ============================================================
// API Client
// ============================================================

/**
 * Upload two files and create a session.
 */
export async function uploadFiles(fileA: File, fileB: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file_a", fileA);
  formData.append("file_b", fileB);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new OverlayApiError(
      response.status,
      errorData.error?.code || "UPLOAD_FAILED",
      errorData.error?.message || `Upload failed with status ${response.status}`,
      errorData.error?.details
    );
  }

  return response.json();
}

/**
 * Generate preview for a file in a session.
 */
export async function generatePreview(
  sessionId: string,
  which: "A" | "B"
): Promise<PreviewResponse> {
  const response = await fetch(
    `${API_BASE_URL}/sessions/${sessionId}/preview?which=${which}`,
    { method: "GET" }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new OverlayApiError(
      response.status,
      errorData.detail?.code || "PREVIEW_FAILED",
      errorData.detail?.message || `Preview generation failed`,
      errorData.detail?.details
    );
  }

  return response.json();
}

/**
 * Compose layers from a session.
 * 
 * This is the main API call - returns layers[] + state.json.
 * NEVER returns a flattened overlay image.
 */
export async function composeLayers(
  sessionId: string,
  options?: ComposeRequest
): Promise<ComposeResponse> {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/compose`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(options || { auto: true }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new OverlayApiError(
      response.status,
      errorData.detail?.code || "COMPOSE_FAILED",
      errorData.detail?.message || `Compose failed with status ${response.status}`,
      errorData.detail?.details
    );
  }

  const data = await response.json();
  
  // Validate response
  const parsed = ComposeResponseSchema.safeParse(data);
  if (!parsed.success) {
    console.warn("Compose response validation failed:", parsed.error);
  }
  
  // Transform snake_case bounds to camelCase for frontend
  const result = parsed.success ? parsed.data : data;
  if (result.bounds) {
    result.bounds = {
      contentBounds: result.bounds.content_bounds,
      geometryBounds: result.bounds.geometry_bounds,
    };
  }
  
  return result as ComposeResponse;
}

/**
 * Update composition state on the backend.
 */
export async function updateCompositionState(
  sessionId: string,
  state: CompositionState
): Promise<ComposeResponse> {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/compose/update-state`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(state),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new OverlayApiError(
      response.status,
      errorData.detail?.code || "UPDATE_FAILED",
      errorData.detail?.message || `Update failed`,
      errorData.detail?.details
    );
  }

  return response.json();
}

/**
 * Get image URL for a preview.
 */
export function getPreviewImageUrl(imageId: string): string {
  return `${API_BASE_URL}/images/${imageId}`;
}

// ============================================================
// Differences API
// ============================================================

export interface RegionInfo {
  id: number;
  bbox: number[];
  area: number;
  centroid: number[];
}

export interface DifferencesResponse {
  session_id: string;
  summary: string;
  regions: RegionInfo[];
  total_regions: number;
  diff_percentage: number;
  ai_available: boolean;
  model_name?: string | null;
  model_display_name?: string | null;
  is_vlm: boolean;
  processing_time_ms: number;
  generated_at: string;
  aligned: boolean;
  alignment_source: string;
  warnings: string[];
}

export interface DifferencesTransform {
  scale: number;
  rotation_deg: number;
  translate_x: number;
  translate_y: number;
}

export interface DifferencesRequest {
  transform?: DifferencesTransform;  // Legacy: B's transform only (A at identity)
  transform_a?: DifferencesTransform;  // Layer A's current transform
  transform_b?: DifferencesTransform;  // Layer B's current transform
}

/**
 * Generate AI-powered differences summary for a session.
 * 
 * @param sessionId - The session ID
 * @param transformA - Layer A's current transform (optional, for proper relative calculation)
 * @param transformB - Layer B's current transform (optional, for proper relative calculation)
 * 
 * When both transformA and transformB are provided, the backend computes the relative
 * transform (B in A's coordinate space) to properly handle cases where the user has
 * manually adjusted either or both layers.
 */
export async function generateDifferences(
  sessionId: string,
  transformA?: DifferencesTransform,
  transformB?: DifferencesTransform
): Promise<DifferencesResponse> {
  const body: DifferencesRequest = {};
  
  if (transformA && transformB) {
    // New mode: Send both transforms for proper relative calculation
    body.transform_a = transformA;
    body.transform_b = transformB;
  } else if (transformB) {
    // Legacy fallback: Only B provided (A at identity)
    body.transform = transformB;
  }
  // If neither provided, backend will use session alignment or identity

  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/differences`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new OverlayApiError(
      response.status,
      errorData.detail?.code || "DIFFERENCES_FAILED",
      errorData.detail?.message || `Failed to generate differences`,
      errorData.detail?.details
    );
  }

  return response.json();
}

// ============================================================
// Session Info API
// ============================================================

export interface FileUploadInfo {
  id: string;
  filename: string;
  size_bytes: number;
  uploaded_at: string;
}

export interface SessionInfoResponse {
  session_id: string;
  status: string;
  file_a: FileUploadInfo | null;
  file_b: FileUploadInfo | null;
}

/**
 * Get session information including file names.
 */
export async function getSessionInfo(sessionId: string): Promise<SessionInfoResponse> {
  const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
    method: "GET",
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new OverlayApiError(
      response.status,
      errorData.detail?.code || "SESSION_FETCH_FAILED",
      errorData.detail?.message || `Failed to fetch session info`,
      errorData.detail?.details
    );
  }

  return response.json();
}
