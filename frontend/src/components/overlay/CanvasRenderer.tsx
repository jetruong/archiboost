"use client";

/**
 * Canvas Renderer Component
 * 
 * Central canvas area that renders all layers:
 * - Composites layers in order with transforms
 * - Handles pan/zoom
 * - Drag interaction for selected layer
 * 
 * CANVAS BOUNDS ARCHITECTURE:
 * 
 * contentBounds: The bounding box that includes ALL visible content from both layers
 *   (geometry, text, callouts, dimensions, etc.). Used for editor canvas sizing.
 *   The editor canvas NEVER clips content - it auto-extends to fit everything.
 * 
 * geometryBounds: The bounding box that includes ONLY geometry/linework, excluding
 *   text and annotations. Used for export cropping to reduce excessive whitespace.
 *   Computed by analyzing rendered pixels and filtering out sparse edge regions.
 */

import React, { useRef, useEffect, useState, useCallback, useMemo } from "react";
import {
  LayerInfo,
  LayerState,
  CanvasState,
  Transform2D,
  BoundingBox,
  getLayerImageSrc,
  transformToCSS,
  blendModeToCSS,
  computeWorkspaceCanvas,
  WORKSPACE_MARGIN_PX,
  EXPORT_CROP_PADDING_PX,
  WHITE_THRESHOLD,
  RegionVisibilityState,
} from "@/lib/overlay-types";
import { RegionInfo } from "@/lib/overlay-api";

interface CanvasRendererProps {
  layers: LayerInfo[];
  layerStates: LayerState[];
  canvas: CanvasState;
  selectedLayerId: string | null;
  zoom: number;
  panX: number;
  panY: number;
  onSelectLayer: (id: string | null) => void;
  onBeginTransform?: (id: string) => void;
  onUpdateTransform: (id: string, transform: Transform2D) => void;
  onCommitTransform?: (id: string, transform: Transform2D) => void;
  onZoomChange: (zoom: number) => void;
  onPanChange: (x: number, y: number) => void;
  // Region annotation props
  regions?: RegionInfo[];
  regionVisibility?: RegionVisibilityState;
}

// Cache for loaded images
const imageCache = new Map<string, HTMLImageElement>();

// Color definitions for dynamic colorization
// The backend produces white linework, frontend colorizes to these RGB values
const COLOR_RGB: Record<string, [number, number, number]> = {
  red: [255, 0, 0],
  cyan: [0, 255, 255],
  blue: [0, 0, 255],
  green: [0, 255, 0],
  magenta: [255, 0, 255],
  yellow: [255, 255, 0],
  black: [0, 0, 0],
  white: [255, 255, 255],
  // Hex colors
  "#ff0000": [255, 0, 0],
  "#00ffff": [0, 255, 255],
  "#0000ff": [0, 0, 255],
  "#00ff00": [0, 255, 0],
  "#ff00ff": [255, 0, 255],
  "#ffff00": [255, 255, 0],
  "#000000": [0, 0, 0],
  "#ffffff": [255, 255, 255],
};

/**
 * Parse a hex color string to RGB values
 */
function hexToRgb(hex: string): [number, number, number] | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [
        parseInt(result[1], 16),
        parseInt(result[2], 16),
        parseInt(result[3], 16),
      ]
    : null;
}

/**
 * Convert RGB to hex color string
 */
function rgbToHex(r: number, g: number, b: number): string {
  return `#${[r, g, b].map(x => x.toString(16).padStart(2, '0')).join('')}`;
}

/**
 * Compute a contrasting color that stands out from two given colors.
 * Uses a simple algorithm: finds a color that maximizes distance from both input colors.
 */
function computeContrastingColor(
  color1: string | null | undefined,
  color2: string | null | undefined
): string {
  // Default colors if not provided
  const c1 = getColorRgb(color1 || "red");
  const c2 = getColorRgb(color2 || "cyan");
  
  // Candidate colors to consider (colors that typically contrast well)
  const candidates: [number, number, number][] = [
    [255, 255, 0],   // Yellow
    [0, 255, 0],     // Green
    [255, 165, 0],   // Orange
    [255, 0, 255],   // Magenta
    [128, 0, 128],   // Purple
    [0, 128, 0],     // Dark Green
    [255, 255, 255], // White
    [0, 0, 0],       // Black
  ];
  
  // Calculate color distance (simple Euclidean in RGB space)
  const colorDistance = (a: [number, number, number], b: [number, number, number]): number => {
    return Math.sqrt(
      Math.pow(a[0] - b[0], 2) +
      Math.pow(a[1] - b[1], 2) +
      Math.pow(a[2] - b[2], 2)
    );
  };
  
  // Find the candidate with maximum minimum distance to both colors
  let bestCandidate = candidates[0];
  let bestScore = 0;
  
  for (const candidate of candidates) {
    const dist1 = colorDistance(candidate, c1);
    const dist2 = colorDistance(candidate, c2);
    // Score is the minimum distance to either color (we want this to be high)
    const score = Math.min(dist1, dist2);
    
    if (score > bestScore) {
      bestScore = score;
      bestCandidate = candidate;
    }
  }
  
  return rgbToHex(bestCandidate[0], bestCandidate[1], bestCandidate[2]);
}

/**
 * Get RGB values for a color (name or hex)
 */
function getColorRgb(color: string | null | undefined): [number, number, number] {
  if (!color) return [255, 255, 255]; // Default to white (no tint)
  
  const lowerColor = color.toLowerCase();
  
  // Check predefined colors
  if (COLOR_RGB[lowerColor]) {
    return COLOR_RGB[lowerColor];
  }
  
  // Try to parse as hex
  const rgb = hexToRgb(lowerColor);
  if (rgb) return rgb;
  
  // Default to white
  return [255, 255, 255];
}

/**
 * ColorizedImage component - renders a white-on-transparent image
 * with dynamic color tinting using canvas.
 */
function ColorizedImage({
  src,
  color,
  className,
  style,
  alt,
}: {
  src: string;
  color: string | null | undefined;
  className?: string;
  style?: React.CSSProperties;
  alt?: string;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = React.useState({ width: 0, height: 0 });
  
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !src) return;
    
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    
    const img = new Image();
    img.crossOrigin = "anonymous";
    
    img.onload = () => {
      // Set canvas dimensions to match image
      canvas.width = img.width;
      canvas.height = img.height;
      setDimensions({ width: img.width, height: img.height });
      
      // Draw original image
      ctx.drawImage(img, 0, 0);
      
      // Get pixel data
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      
      // Get target color
      const [r, g, b] = getColorRgb(color);
      
      // Colorize: convert white pixels to target color
      // White pixels have R=G=B=255, we preserve the alpha
      for (let i = 0; i < data.length; i += 4) {
        const alpha = data[i + 3];
        if (alpha > 0) {
          // This pixel is visible, colorize it
          // The linework is white, so we just set the color
          data[i] = r;     // R
          data[i + 1] = g; // G
          data[i + 2] = b; // B
          // Keep alpha unchanged
        }
      }
      
      // Put modified data back
      ctx.putImageData(imageData, 0, 0);
    };
    
    img.src = src;
  }, [src, color]);
  
  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        ...style,
        width: "100%",
        height: "100%",
      }}
      aria-label={alt}
    />
  );
}

export function CanvasRenderer({
  layers,
  layerStates,
  canvas,
  selectedLayerId,
  zoom,
  panX,
  panY,
  onSelectLayer,
  onBeginTransform,
  onUpdateTransform,
  onCommitTransform,
  onZoomChange,
  onPanChange,
  regions = [],
  regionVisibility = {},
}: CanvasRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loadedImages, setLoadedImages] = useState<Map<string, HTMLImageElement>>(new Map());
  
  // Drag state
  const [isDragging, setIsDragging] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [originalTransform, setOriginalTransform] = useState<Transform2D | null>(null);
  const lastTransformRef = useRef<Transform2D | null>(null);
  const didBeginTransformRef = useRef(false);

  // ============================================================
  // FIXED WORKSPACE CANVAS
  // The editor ALWAYS shows a generous white workspace - NO clipping.
  // Uses fixed margin (400px) on all sides, NOT dynamic sizing.
  // Cropping only happens during export.
  // ============================================================
  
  const workspaceCanvas = useMemo(() => {
    return computeWorkspaceCanvas(layers, WORKSPACE_MARGIN_PX);
  }, [layers]);
  
  // Alias for backwards compatibility with existing render code
  const extendedCanvasBounds = useMemo(() => ({
    width: workspaceCanvas.width,
    height: workspaceCanvas.height,
    offsetX: workspaceCanvas.offsetX,
    offsetY: workspaceCanvas.offsetY,
  }), [workspaceCanvas]);

  // Load images on mount and when layers change
  useEffect(() => {
    let cancelled = false;

    const loadImages = async () => {
      const newImages = new Map<string, HTMLImageElement>();
      
      for (const layer of layers) {
        const src = getLayerImageSrc(layer);
        if (!src) continue;
        
        // Check cache first
        if (imageCache.has(layer.id)) {
          newImages.set(layer.id, imageCache.get(layer.id)!);
          continue;
        }
        
        // Load image
        try {
          const img = new Image();
          await new Promise<void>((resolve, reject) => {
            img.onload = () => resolve();
            img.onerror = reject;
            img.src = src;
          });
          
          imageCache.set(layer.id, img);
          newImages.set(layer.id, img);
        } catch (err) {
          console.error(`Failed to load image for layer ${layer.id}:`, err);
        }
      }
      
      if (!cancelled) {
        setLoadedImages(newImages);
      }
    };
    
    loadImages();

    return () => {
      cancelled = true;
    };
  }, [layers]);

  // Handle wheel zoom
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.min(Math.max(zoom * delta, 0.1), 10);
    
    onZoomChange(newZoom);
  }, [zoom, onZoomChange]);

  // Handle mouse down
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    const isCanvasBackground = target.classList.contains("canvas-bg") || 
                               target === containerRef.current ||
                               target.classList.contains("workspace-bg");
    
    // Middle click or Alt + left click = always pan
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      e.preventDefault();
      setIsPanning(true);
      setDragStart({ x: e.clientX - panX, y: e.clientY - panY });
      return;
    }
    
    // Left click on canvas background = pan (drag to move workspace)
    if (e.button === 0 && isCanvasBackground) {
      e.preventDefault();
      setIsPanning(true);
      setDragStart({ x: e.clientX - panX, y: e.clientY - panY });
      return;
    }
    
    // Left click on selected layer = drag the layer
    if (e.button === 0 && selectedLayerId) {
      const state = layerStates.find((s) => s.id === selectedLayerId);
      if (state && !state.locked) {
        setIsDragging(true);
        setDragStart({ x: e.clientX, y: e.clientY });
        setOriginalTransform({ ...state.transform });
        lastTransformRef.current = { ...state.transform };
        didBeginTransformRef.current = false;
      }
    }
  }, [selectedLayerId, layerStates, panX, panY]);

  // Handle mouse move
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isPanning) {
      onPanChange(e.clientX - dragStart.x, e.clientY - dragStart.y);
      return;
    }
    
    if (isDragging && selectedLayerId && originalTransform) {
      const dx = (e.clientX - dragStart.x) / zoom;
      const dy = (e.clientY - dragStart.y) / zoom;

      if (!didBeginTransformRef.current && (dx !== 0 || dy !== 0)) {
        onBeginTransform?.(selectedLayerId);
        didBeginTransformRef.current = true;
      }
      
      const newTransform: Transform2D = {
        ...originalTransform,
        translate_x: originalTransform.translate_x + dx,
        translate_y: originalTransform.translate_y + dy,
      };
      
      lastTransformRef.current = newTransform;
      onUpdateTransform(selectedLayerId, newTransform);
    }
  }, [
    isPanning,
    isDragging,
    selectedLayerId,
    originalTransform,
    dragStart,
    zoom,
    onPanChange,
    onBeginTransform,
    onUpdateTransform,
  ]);

  // Handle mouse up
  const handleMouseUp = useCallback(() => {
    if (isDragging && selectedLayerId && onCommitTransform && lastTransformRef.current) {
      onCommitTransform(selectedLayerId, lastTransformRef.current);
    }
    setIsDragging(false);
    setIsPanning(false);
    setOriginalTransform(null);
    lastTransformRef.current = null;
    didBeginTransformRef.current = false;
  }, [isDragging, selectedLayerId, onCommitTransform]);

  // Handle click on canvas background
  const handleCanvasClick = useCallback((e: React.MouseEvent) => {
    // Deselect if clicking on background
    if (e.target === containerRef.current || (e.target as HTMLElement).classList.contains("canvas-bg")) {
      onSelectLayer(null);
    }
  }, [onSelectLayer]);

  // Get layer state by ID
  const getState = (id: string) => layerStates.find((s) => s.id === id);
  const getInfo = (id: string) => layers.find((l) => l.id === id);

  // Build render order (layer states determine order)
  const renderOrder = layerStates
    .map((state) => ({
      state,
      info: getInfo(state.id),
      image: loadedImages.get(state.id),
    }))
    .filter((item) => item.info && item.image && item.state.visible);

  return (
    <div
      ref={containerRef}
      className="flex-1 bg-gray-100 overflow-hidden cursor-grab active:cursor-grabbing relative workspace-bg"
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onClick={handleCanvasClick}
    >
      {/* Zoom indicator */}
      <div className="absolute top-4 left-4 z-10 bg-white/90 backdrop-blur border border-gray-200 px-3 py-1 rounded text-xs text-gray-600 shadow-sm">
        {Math.round(zoom * 100)}%
      </div>

      {/* Layer container with pan/zoom - no white canvas box, layers float on gray */}
      <div
        className="absolute inset-0 flex items-center justify-center canvas-bg"
        style={{
          transform: `translate(${panX}px, ${panY}px) scale(${zoom})`,
          transformOrigin: "center center",
        }}
      >
        {/* Layers render directly - no background box */}
        <div
          className="relative"
          style={{
            // Size to contain all content but no visible background
            width: extendedCanvasBounds.width,
            height: extendedCanvasBounds.height,
          }}
        >
          {/* Render all layers in order, with offset to handle negative coordinates */}
          {renderOrder.map(({ state, info, image }) => {
            if (!info || !image) return null;
            
            const isSelected = state.id === selectedLayerId;
            const blendMode = blendModeToCSS(state.blend_mode);
            
            // Apply offset to account for content that extends into negative coordinates
            // This ensures all content remains visible within the extended canvas
            const adjustedTransform: Transform2D = {
              ...state.transform,
              translate_x: state.transform.translate_x + extendedCanvasBounds.offsetX,
              translate_y: state.transform.translate_y + extendedCanvasBounds.offsetY,
            };
            const transform = transformToCSS(adjustedTransform);
            
            return (
              <div
                key={state.id}
                className={`absolute top-0 left-0 ${isSelected ? "ring-2 ring-blue-500 ring-offset-2 ring-offset-gray-100" : ""}`}
                style={{
                  width: info.width,
                  height: info.height,
                  transform,
                  transformOrigin: `${state.transform.pivot_x * 100}% ${state.transform.pivot_y * 100}%`,
                  opacity: state.opacity,
                  mixBlendMode: blendMode as any,
                  pointerEvents: state.locked ? "none" : "auto",
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectLayer(state.id);
                }}
              >
                <ColorizedImage
                  src={getLayerImageSrc(info) || ""}
                  color={state.color}
                  alt={info.name}
                  className="w-full h-full"
                />
              </div>
            );
          })}
          
          {/* Selection handles for selected layer */}
          {selectedLayerId && (() => {
            const state = getState(selectedLayerId);
            const info = getInfo(selectedLayerId);
            if (!state || !info || state.locked) return null;
            
            // Apply same offset as layers for selection handles
            const adjustedTransform: Transform2D = {
              ...state.transform,
              translate_x: state.transform.translate_x + extendedCanvasBounds.offsetX,
              translate_y: state.transform.translate_y + extendedCanvasBounds.offsetY,
            };
            
            return (
              <div
                className="absolute pointer-events-none"
                style={{
                  width: info.width,
                  height: info.height,
                  transform: transformToCSS(adjustedTransform),
                  transformOrigin: `${state.transform.pivot_x * 100}% ${state.transform.pivot_y * 100}%`,
                }}
              >
                {/* Corner handles */}
                <div className="absolute -top-1 -left-1 w-3 h-3 bg-white border-2 border-blueprint-500 pointer-events-auto cursor-nw-resize" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-white border-2 border-blueprint-500 pointer-events-auto cursor-ne-resize" />
                <div className="absolute -bottom-1 -left-1 w-3 h-3 bg-white border-2 border-blueprint-500 pointer-events-auto cursor-sw-resize" />
                <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-white border-2 border-blueprint-500 pointer-events-auto cursor-se-resize" />
                
                {/* Rotation handle */}
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 w-4 h-4 bg-white border-2 border-blueprint-500 rounded-full pointer-events-auto cursor-grab" />
              </div>
            );
          })()}

          {/* Region annotation bounding boxes */}
          {(() => {
            // Compute contrasting color based on layer colors
            const layerColors = layerStates
              .filter(s => s.visible)
              .map(s => s.color)
              .filter(Boolean);
            
            const bboxColor = computeContrastingColor(
              layerColors[0] || "red",
              layerColors[1] || "cyan"
            );
            
            // Compute a fill color (lighter version or complementary)
            const bboxRgb = hexToRgb(bboxColor) || [255, 255, 0];
            const fillColor = `rgba(${bboxRgb[0]}, ${bboxRgb[1]}, ${bboxRgb[2]}, 0.15)`;
            
            return regions.map((region) => {
              const isVisible = regionVisibility[region.id] ?? true;
              if (!isVisible) return null;
              
              const [x, y, width, height] = region.bbox;
              
              return (
                <div
                  key={`region-${region.id}`}
                  className="absolute pointer-events-none"
                  style={{
                    left: x + extendedCanvasBounds.offsetX,
                    top: y + extendedCanvasBounds.offsetY,
                    width,
                    height,
                  }}
                >
                  {/* Bounding box */}
                  <div 
                    className="absolute inset-0 border-2"
                    style={{
                      borderColor: bboxColor,
                      boxShadow: "0 0 0 1px rgba(255,255,255,0.5)",
                    }}
                  />
                  
                  {/* Semi-transparent fill */}
                  <div 
                    className="absolute inset-0"
                    style={{ backgroundColor: fillColor }}
                  />
                  
                  {/* Label */}
                  <div 
                    className="absolute -top-6 left-0 px-1.5 py-0.5 text-white text-xs font-bold rounded shadow"
                    style={{ 
                      minWidth: "24px", 
                      textAlign: "center",
                      backgroundColor: bboxColor,
                    }}
                  >
                    R{region.id}
                  </div>
                </div>
              );
            });
          })()}
        </div>
      </div>

      {/* Keyboard shortcuts hint */}
      <div className="absolute bottom-4 right-4 z-10 bg-white/90 backdrop-blur border border-gray-200 px-3 py-2 rounded text-xs text-gray-500 shadow-sm">
        <div>Scroll: Zoom | Drag: Pan</div>
        <div>Arrow keys: Nudge (Shift: 10px)</div>
      </div>
    </div>
  );
}

/**
 * Compute tight bounding box of non-white pixels in image data.
 * Used for export cropping to remove excessive whitespace.
 */
function computeNonWhiteBounds(
  imageData: ImageData,
  whiteThreshold: number = WHITE_THRESHOLD,
  padding: number = EXPORT_CROP_PADDING_PX
): BoundingBox | null {
  const { width, height, data } = imageData;
  
  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  
  // Scan for non-white pixels
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      // If any channel is below threshold, it's not white
      if (r < whiteThreshold || g < whiteThreshold || b < whiteThreshold) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  
  // No content found
  if (maxX < minX || maxY < minY) {
    return null;
  }
  
  // Add padding and clamp to image bounds
  const finalMinX = Math.max(0, minX - padding);
  const finalMinY = Math.max(0, minY - padding);
  const finalMaxX = Math.min(width - 1, maxX + padding);
  const finalMaxY = Math.min(height - 1, maxY + padding);
  
  return {
    x: finalMinX,
    y: finalMinY,
    width: finalMaxX - finalMinX + 1,
    height: finalMaxY - finalMinY + 1,
  };
}

/**
 * Export canvas to PNG with automatic cropping to non-white pixels.
 * 
 * EXPORT BEHAVIOR:
 * 1. Renders the full workspace canvas (same as editor shows)
 * 2. Scans for non-white pixels to find content bounds
 * 3. Crops to those bounds + small padding
 * 4. Exports the cropped PNG
 * 
 * This produces a tight output image without excessive whitespace,
 * while the editor always shows the full generous workspace.
 */
export async function exportCanvasToPNG(
  layers: LayerInfo[],
  layerStates: LayerState[],
  canvas: CanvasState
): Promise<Blob> {
  // Step 1: Compute workspace canvas (same as editor)
  const workspace = computeWorkspaceCanvas(layers, WORKSPACE_MARGIN_PX);
  
  // Create offscreen canvas matching the workspace
  const offscreen = document.createElement("canvas");
  offscreen.width = workspace.width;
  offscreen.height = workspace.height;
  
  const ctx = offscreen.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Failed to get canvas context");
  
  // Fill background white
  ctx.fillStyle = canvas.background_color || "#FFFFFF";
  ctx.fillRect(0, 0, workspace.width, workspace.height);
  
  // Step 2: Render all visible layers (centered in workspace)
  for (const state of layerStates) {
    if (!state.visible) continue;
    
    const info = layers.find((l) => l.id === state.id);
    if (!info) continue;
    
    const src = getLayerImageSrc(info);
    if (!src) continue;
    
    // Load image
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = src;
    });
    
    // Apply transform with workspace offset (content is centered)
    ctx.save();
    ctx.globalAlpha = state.opacity;
    ctx.globalCompositeOperation = state.blend_mode as GlobalCompositeOperation;
    
    const t = state.transform;
    const pivotX = t.pivot_x * info.width;
    const pivotY = t.pivot_y * info.height;
    
    // Workspace offset centers content, then apply user transform
    ctx.translate(
      workspace.offsetX + t.translate_x + pivotX,
      workspace.offsetY + t.translate_y + pivotY
    );
    ctx.rotate((t.rotation_deg * Math.PI) / 180);
    ctx.scale(t.scale_x, t.scale_y);
    ctx.translate(-pivotX, -pivotY);
    
    ctx.drawImage(img, 0, 0);
    ctx.restore();
  }
  
  // Step 3: Find non-white content bounds for cropping
  const imageData = ctx.getImageData(0, 0, workspace.width, workspace.height);
  const contentBounds = computeNonWhiteBounds(imageData);
  
  // Step 4: Crop to content bounds and export
  if (contentBounds && contentBounds.width > 0 && contentBounds.height > 0) {
    // Create cropped canvas
    const croppedCanvas = document.createElement("canvas");
    croppedCanvas.width = contentBounds.width;
    croppedCanvas.height = contentBounds.height;
    
    const croppedCtx = croppedCanvas.getContext("2d");
    if (!croppedCtx) throw new Error("Failed to get cropped canvas context");
    
    // Copy just the content region
    croppedCtx.drawImage(
      offscreen,
      contentBounds.x, contentBounds.y, contentBounds.width, contentBounds.height,
      0, 0, contentBounds.width, contentBounds.height
    );
    
    // Export cropped canvas
    return new Promise((resolve, reject) => {
      croppedCanvas.toBlob(
        (blob) => {
          if (blob) resolve(blob);
          else reject(new Error("Failed to export cropped canvas"));
        },
        "image/png"
      );
    });
  }
  
  // Fallback: export full workspace if no content found
  return new Promise((resolve, reject) => {
    offscreen.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new Error("Failed to export canvas"));
      },
      "image/png"
    );
  });
}
