/**
 * Types for the layer-based overlay system.
 * 
 * These types mirror the backend models for TypeScript type safety.
 */

// ============================================================
// Constants for Canvas/Workspace Bounds
// ============================================================

/**
 * WORKSPACE_MARGIN_PX: Fixed margin added to all sides of the workspace canvas.
 * The editor always shows a generous white workspace beyond the drawing content.
 * This is NOT dynamic - it's a fixed constant for predictable behavior.
 */
export const WORKSPACE_MARGIN_PX = 400;

/**
 * EXPORT_CROP_PADDING_PX: Small padding added around content when exporting.
 * Export crops to non-white pixels + this padding.
 */
export const EXPORT_CROP_PADDING_PX = 20;

/**
 * WHITE_THRESHOLD: Pixel value above which is considered "white" for export cropping.
 * Pixels with R, G, B all > this value are treated as background.
 */
export const WHITE_THRESHOLD = 245;

/**
 * Legacy constants for backwards compatibility
 * @deprecated Use WORKSPACE_MARGIN_PX instead
 */
export const CANVAS_PADDING_PX = WORKSPACE_MARGIN_PX;
export const CANVAS_PADDING_MIN_PX = 64;
export const EXPORT_MARGIN_PX = EXPORT_CROP_PADDING_PX;

// ============================================================
// Bounding Box Types
// ============================================================

/**
 * BoundingBox represents a rectangular region.
 */
export interface BoundingBox {
  x: number;      // Left edge
  y: number;      // Top edge
  width: number;  // Width
  height: number; // Height
}

/**
 * contentBounds: Bounding box that includes ALL visible content from both layers
 * (geometry, text, callouts, dimensions, etc.). Used for editor canvas sizing
 * to ensure NOTHING is ever clipped.
 * 
 * geometryBounds: Bounding box that includes ONLY geometry/linework, excluding
 * text and annotations. Used for export cropping to reduce excessive whitespace.
 */
export interface LayerBounds {
  contentBounds: BoundingBox;    // For editor - includes everything
  geometryBounds: BoundingBox;   // For export - geometry only
}

// ============================================================
// Enums
// ============================================================

export type LayerSource = "pair1" | "pair2";
export type LayerType = "base" | "linework" | "text" | "annotation";
export type BlendMode = "normal" | "multiply" | "screen" | "overlay" | "darken" | "lighten";
export type ComposeStatus = "success" | "auto_failed_fallback_manual";

// ============================================================
// Transform
// ============================================================

export interface Transform2D {
  translate_x: number;
  translate_y: number;
  scale_x: number;
  scale_y: number;
  rotation_deg: number;
  pivot_x: number;
  pivot_y: number;
}

export const createIdentityTransform = (): Transform2D => ({
  translate_x: 0,
  translate_y: 0,
  scale_x: 1,
  scale_y: 1,
  rotation_deg: 0,
  pivot_x: 0.5,
  pivot_y: 0.5,
});

// ============================================================
// Layer
// ============================================================

export interface LayerInfo {
  id: string;
  name: string;
  source: LayerSource;
  type: LayerType;
  png_url?: string | null;
  png_base64?: string | null;
  width: number;
  height: number;
  default_color?: string | null;
  default_opacity: number;
}

export interface LayerState {
  id: string;
  visible: boolean;
  opacity: number;
  blend_mode: BlendMode;
  color?: string | null;
  transform: Transform2D;
  locked: boolean;
}

// ============================================================
// Canvas & Composition
// ============================================================

export interface CanvasState {
  width: number;
  height: number;
  dpi: number;
  background_color: string;
  coordinate_system: string;
}

export interface AlignmentState {
  method: string;
  confidence: number;
  transform: Transform2D;
  phase_response?: number | null;
  overlap_ratio?: number | null;
  refinement_method?: string | null;
  warnings: string[];
}

export interface ColorPalette {
  name: string;
  colors: Record<string, string>;
}

export interface CompositionDefaults {
  palette: ColorPalette;
  snap_angles: number[];
  snap_enabled: boolean;
  grid_size: number;
}

export interface CompositionState {
  version: string;
  created_at: string;
  canvas: CanvasState;
  layers: LayerState[];
  alignment: AlignmentState;
  defaults: CompositionDefaults;
}

// ============================================================
// API Request/Response
// ============================================================

export interface ComposeRequest {
  auto?: boolean;
  rotation_constraint?: "NONE" | "RIGHT_ANGLES" | "SMALL_ANGLE";
  debug?: boolean;
  manual_transform?: Transform2D;
}

export interface ComposeResponse {
  session_id: string;
  status: ComposeStatus;
  confidence: number;
  layers: LayerInfo[];
  state: CompositionState;
  // Bounds information for canvas sizing and export cropping
  bounds?: LayerBounds | null;
  warnings: string[];
  reason?: string | null;
  processing_time_ms: number;
  generated_at: string;
}

// ============================================================
// Editor State
// ============================================================

export interface EditorState {
  // Current layer/composition state
  layers: LayerInfo[];
  layerStates: LayerState[];
  canvas: CanvasState;
  alignment: AlignmentState;
  
  // Selection
  selectedLayerId: string | null;
  
  // View state
  zoom: number;
  panX: number;
  panY: number;
  
  // Undo/redo
  history: LayerState[][];
  historyIndex: number;
  
  // Status
  isDirty: boolean;
  isLoading: boolean;
  error: string | null;
}

export const createInitialEditorState = (): EditorState => ({
  layers: [],
  layerStates: [],
  canvas: {
    width: 800,
    height: 600,
    dpi: 250,
    background_color: "#FFFFFF",
    coordinate_system: "top_left_origin",
  },
  alignment: {
    method: "identity",
    confidence: 0,
    transform: createIdentityTransform(),
    warnings: [],
  },
  selectedLayerId: null,
  zoom: 1,
  panX: 0,
  panY: 0,
  history: [],
  historyIndex: 0,
  isDirty: false,
  isLoading: false,
  error: null,
});

// ============================================================
// Utility Functions
// ============================================================

/**
 * Get the image source (URL or base64) for a layer.
 */
export function getLayerImageSrc(layer: LayerInfo): string | null {
  if (layer.png_base64) return layer.png_base64;
  if (layer.png_url) return layer.png_url;
  return null;
}

/**
 * Convert transform to CSS transform string.
 */
export function transformToCSS(t: Transform2D): string {
  const parts: string[] = [];
  
  if (t.translate_x !== 0 || t.translate_y !== 0) {
    parts.push(`translate(${t.translate_x}px, ${t.translate_y}px)`);
  }
  
  if (t.rotation_deg !== 0) {
    parts.push(`rotate(${t.rotation_deg}deg)`);
  }
  
  if (t.scale_x !== 1 || t.scale_y !== 1) {
    parts.push(`scale(${t.scale_x}, ${t.scale_y})`);
  }
  
  return parts.length > 0 ? parts.join(" ") : "none";
}

/**
 * Convert blend mode to CSS mix-blend-mode value.
 */
export function blendModeToCSS(mode: BlendMode): string {
  const mapping: Record<BlendMode, string> = {
    normal: "normal",
    multiply: "multiply",
    screen: "screen",
    overlay: "overlay",
    darken: "darken",
    lighten: "lighten",
  };
  return mapping[mode] || "normal";
}

/**
 * Snap angle to nearest snap angle.
 */
export function snapAngle(angle: number, snapAngles: number[]): number {
  const normalized = ((angle % 360) + 360) % 360;
  
  let closest = normalized;
  let minDiff = Infinity;
  
  for (const snap of snapAngles) {
    const diff = Math.abs(normalized - snap);
    const wrapDiff = Math.abs(normalized - snap + 360);
    const wrapDiff2 = Math.abs(normalized - snap - 360);
    const minWrapDiff = Math.min(diff, wrapDiff, wrapDiff2);
    
    if (minWrapDiff < minDiff) {
      minDiff = minWrapDiff;
      closest = snap;
    }
  }
  
  return minDiff < 5 ? closest : normalized; // Snap if within 5 degrees
}

// ============================================================
// Bounding Box Computation Utilities
// ============================================================

/**
 * Compute the transformed bounding box of a layer.
 * Takes the layer's original dimensions and its transform, and returns
 * the axis-aligned bounding box that contains the transformed layer.
 */
export function computeTransformedBounds(
  width: number,
  height: number,
  transform: Transform2D
): BoundingBox {
  // Get corner points of the original rectangle
  const corners = [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height },
  ];
  
  // Apply transform to each corner
  const pivotX = transform.pivot_x * width;
  const pivotY = transform.pivot_y * height;
  const radians = (transform.rotation_deg * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  
  const transformedCorners = corners.map(({ x, y }) => {
    // Translate to pivot
    const px = x - pivotX;
    const py = y - pivotY;
    
    // Scale
    const sx = px * transform.scale_x;
    const sy = py * transform.scale_y;
    
    // Rotate
    const rx = sx * cos - sy * sin;
    const ry = sx * sin + sy * cos;
    
    // Translate back from pivot and apply translation
    return {
      x: rx + pivotX + transform.translate_x,
      y: ry + pivotY + transform.translate_y,
    };
  });
  
  // Find axis-aligned bounding box
  const xs = transformedCorners.map(p => p.x);
  const ys = transformedCorners.map(p => p.y);
  
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  
  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

/**
 * Union two bounding boxes into a single bounding box that contains both.
 */
export function unionBounds(a: BoundingBox, b: BoundingBox): BoundingBox {
  const minX = Math.min(a.x, b.x);
  const minY = Math.min(a.y, b.y);
  const maxX = Math.max(a.x + a.width, b.x + b.width);
  const maxY = Math.max(a.y + a.height, b.y + b.height);
  
  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  };
}

/**
 * Add symmetric padding to a bounding box.
 */
export function addPadding(bbox: BoundingBox, padding: number): BoundingBox {
  return {
    x: bbox.x - padding,
    y: bbox.y - padding,
    width: bbox.width + padding * 2,
    height: bbox.height + padding * 2,
  };
}

/**
 * Workspace canvas configuration for the editor.
 */
export interface WorkspaceCanvas {
  /** Total workspace width in pixels */
  width: number;
  /** Total workspace height in pixels */
  height: number;
  /** X offset to center content in workspace */
  offsetX: number;
  /** Y offset to center content in workspace */
  offsetY: number;
  /** The base content width (max of layer widths) */
  baseWidth: number;
  /** The base content height (max of layer heights) */
  baseHeight: number;
}

/**
 * Compute a fixed workspace canvas that is larger than the content.
 * 
 * The editor ALWAYS shows a generous white workspace - no dynamic cropping.
 * Cropping only happens during export.
 * 
 * @param layers - Array of layer info (to get dimensions)
 * @param margin - Fixed margin on all sides (default: WORKSPACE_MARGIN_PX)
 * @returns WorkspaceCanvas with dimensions and centering offsets
 */
export function computeWorkspaceCanvas(
  layers: LayerInfo[],
  margin: number = WORKSPACE_MARGIN_PX
): WorkspaceCanvas {
  // Find max dimensions across all layers
  let maxWidth = 800;  // Default minimum
  let maxHeight = 600;
  
  for (const layer of layers) {
    maxWidth = Math.max(maxWidth, layer.width);
    maxHeight = Math.max(maxHeight, layer.height);
  }
  
  // Workspace is content + margin on all sides
  const canvasWidth = maxWidth + 2 * margin;
  const canvasHeight = maxHeight + 2 * margin;
  
  return {
    width: canvasWidth,
    height: canvasHeight,
    offsetX: margin,  // Content starts after the left margin
    offsetY: margin,  // Content starts after the top margin
    baseWidth: maxWidth,
    baseHeight: maxHeight,
  };
}

/**
 * Legacy function for backwards compatibility.
 * @deprecated Use computeWorkspaceCanvas instead
 */
export function computeContentBounds(
  layers: LayerInfo[],
  layerStates: LayerState[],
  padding?: number
): BoundingBox {
  const workspace = computeWorkspaceCanvas(layers, padding ?? WORKSPACE_MARGIN_PX);
  return {
    x: 0,
    y: 0,
    width: workspace.width,
    height: workspace.height,
  };
}

// ============================================================
// Region Annotation Types
// ============================================================

/**
 * A detected difference region from the analysis.
 */
export interface RegionInfo {
  id: number;
  bbox: number[];  // [x, y, width, height]
  area: number;
  centroid: number[];  // [x, y]
}

/**
 * Visibility state for a region annotation.
 */
export interface RegionVisibilityState {
  [regionId: number]: boolean;
}

/**
 * Create initial visibility state (all visible) for regions.
 */
export function createRegionVisibilityState(regions: RegionInfo[]): RegionVisibilityState {
  const state: RegionVisibilityState = {};
  for (const region of regions) {
    state[region.id] = true;
  }
  return state;
}
