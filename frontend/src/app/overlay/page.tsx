"use client";

/**
 * Overlay Editor Page
 * 
 * Main page for the layer-based overlay editor.
 * This page loads composition data from the API and provides
 * an Adobe-like interface for viewing and editing overlays.
 * 
 * Key principles:
 * - Backend provides layers[] + state.json (NEVER a flattened image)
 * - Frontend handles all rendering, interaction, and export
 * - Manual adjustment is always possible, even if auto-alignment fails
 */

import { useState, useEffect, useCallback, useReducer, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  LayerInfo,
  LayerState,
  Transform2D,
  CanvasState,
  AlignmentState,
  ComposeStatus,
  createIdentityTransform,
  CompositionState,
  ColorPalette,
  RegionVisibilityState,
  createRegionVisibilityState,
} from "@/lib/overlay-types";
import {
  composeLayers,
  uploadFiles,
  generatePreview,
  generateDifferences,
  getSessionInfo,
  DifferencesResponse,
  DifferencesTransform,
  RegionInfo,
} from "@/lib/overlay-api";
import {
  LayerPanel,
  PropertiesPanel,
  CanvasRenderer,
  Toolbar,
  exportCanvasToPNG,
  RegionAnnotationsPanel,
} from "@/components/overlay";

const HISTORY_LIMIT = 200;

function cloneLayerStates(states: LayerState[]): LayerState[] {
  return states.map((state) => ({
    ...state,
    transform: { ...state.transform },
  }));
}

type LayerHistoryState = {
  past: LayerState[][];
  present: LayerState[];
  future: LayerState[][];
};

type LayerHistoryAction =
  | { type: "INIT"; states: LayerState[] }
  | { type: "CHECKPOINT" }
  | { type: "SET_PRESENT"; updater: (prev: LayerState[]) => LayerState[] }
  | { type: "APPLY"; updater: (prev: LayerState[]) => LayerState[] }
  | { type: "UNDO" }
  | { type: "REDO" };

function layerHistoryReducer(
  state: LayerHistoryState,
  action: LayerHistoryAction
): LayerHistoryState {
  switch (action.type) {
    case "INIT": {
      return {
        past: [],
        present: cloneLayerStates(action.states),
        future: [],
      };
    }
    case "CHECKPOINT": {
      if (state.present.length === 0) return state;
      const nextPast = [...state.past, cloneLayerStates(state.present)].slice(
        -HISTORY_LIMIT
      );
      return {
        past: nextPast,
        present: state.present,
        future: [],
      };
    }
    case "SET_PRESENT": {
      return {
        ...state,
        present: cloneLayerStates(action.updater(state.present)),
      };
    }
    case "APPLY": {
      const nextPresent = cloneLayerStates(action.updater(state.present));
      const nextPast = [...state.past, cloneLayerStates(state.present)].slice(
        -HISTORY_LIMIT
      );
      return {
        past: nextPast,
        present: nextPresent,
        future: [],
      };
    }
    case "UNDO": {
      if (state.past.length === 0) return state;
      const previous = state.past[state.past.length - 1];
      const nextPast = state.past.slice(0, -1);
      return {
        past: nextPast,
        present: cloneLayerStates(previous),
        future: [cloneLayerStates(state.present), ...state.future],
      };
    }
    case "REDO": {
      if (state.future.length === 0) return state;
      const next = state.future[0];
      const nextFuture = state.future.slice(1);
      const nextPast = [...state.past, cloneLayerStates(state.present)].slice(
        -HISTORY_LIMIT
      );
      return {
        past: nextPast,
        present: cloneLayerStates(next),
        future: nextFuture,
      };
    }
    default:
      return state;
  }
}

export default function OverlayPage() {
  return (
    <Suspense fallback={<LoadingScreen message="Loading..." />}>
      <OverlayEditor />
    </Suspense>
  );
}

function OverlayEditor() {
  const searchParams = useSearchParams();

  // Get session ID from URL
  const sessionId = searchParams.get("session");

  // ============================================================
  // State
  // ============================================================

  // API response state
  const [status, setStatus] = useState<ComposeStatus>("success");
  const [confidence, setConfidence] = useState(0);
  const [warnings, setWarnings] = useState<string[]>([]);

  // Layer data
  const [layers, setLayers] = useState<LayerInfo[]>([]);
  const [layerHistory, dispatchLayerHistory] = useReducer(layerHistoryReducer, {
    past: [],
    present: [],
    future: [],
  });
  const layerStates = layerHistory.present;
  const [originalLayerStates, setOriginalLayerStates] = useState<LayerState[]>([]);

  // Canvas state
  const [canvas, setCanvas] = useState<CanvasState>({
    width: 800,
    height: 600,
    dpi: 250,
    background_color: "#FFFFFF",
    coordinate_system: "top_left_origin",
  });

  // Alignment state
  const [alignment, setAlignment] = useState<AlignmentState>({
    method: "identity",
    confidence: 0,
    transform: createIdentityTransform(),
    warnings: [],
  });

  // Palette
  const [palette, setPalette] = useState<ColorPalette>({
    name: "default",
    colors: {
      red: "#FF0000",
      cyan: "#00FFFF",
      blue: "#0000FF",
      green: "#00FF00",
      magenta: "#FF00FF",
      yellow: "#FFFF00",
      black: "#000000",
      white: "#FFFFFF",
    },
  });
  const [snapAngles, setSnapAngles] = useState([0, 90, 180, 270]);

  // Selection
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);

  // View state
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);

  // UI state
  const [isLoading, setIsLoading] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // File names
  const [fileNameA, setFileNameA] = useState<string>("Drawing A");
  const [fileNameB, setFileNameB] = useState<string>("Drawing B");

  // Differences state
  const [isGeneratingDiffs, setIsGeneratingDiffs] = useState(false);
  const [differencesResult, setDifferencesResult] = useState<DifferencesResponse | null>(null);
  const [differencesError, setDifferencesError] = useState<string | null>(null);

  // Region annotations visibility state
  const [regionVisibility, setRegionVisibility] = useState<RegionVisibilityState>({});
  const [allRegionsVisible, setAllRegionsVisible] = useState(true);

  // ============================================================
  // Load composition from API
  // ============================================================

  useEffect(() => {
    if (!sessionId) {
      setError("No session ID provided. Please upload files first.");
      setIsLoading(false);
      return;
    }

    async function loadComposition() {
      try {
        setIsLoading(true);
        setError(null);

        // Fetch session info to get file names
        try {
          const sessionInfo = await getSessionInfo(sessionId!);
          if (sessionInfo.file_a?.filename) {
            setFileNameA(sessionInfo.file_a.filename);
          }
          if (sessionInfo.file_b?.filename) {
            setFileNameB(sessionInfo.file_b.filename);
          }
        } catch (err) {
          console.warn("Failed to fetch session info:", err);
        }

        const response = await composeLayers(sessionId!, { auto: true });

        // Update state from response
        setStatus(response.status);
        setConfidence(response.confidence);
        setWarnings(response.warnings);
        setLayers(response.layers);
        const initialLayerStates = cloneLayerStates(response.state.layers);
        dispatchLayerHistory({ type: "INIT", states: initialLayerStates });
        setOriginalLayerStates(initialLayerStates);
        setCanvas(response.state.canvas);
        setAlignment(response.state.alignment);
        setPalette(response.state.defaults.palette);
        setSnapAngles(response.state.defaults.snap_angles);

        // Auto-fit zoom
        const containerWidth = window.innerWidth - 64 - 72 - 288; // Account for panels
        const containerHeight = window.innerHeight - 48 - 64; // Account for toolbar and header
        const scaleX = containerWidth / response.state.canvas.width;
        const scaleY = containerHeight / response.state.canvas.height;
        const fitZoom = Math.min(scaleX, scaleY, 1) * 0.9;
        setZoom(fitZoom);

      } catch (err) {
        console.error("Failed to load composition:", err);
        setError(err instanceof Error ? err.message : "Failed to load composition");
      } finally {
        setIsLoading(false);
      }
    }

    loadComposition();
  }, [sessionId]);

  // ============================================================
  // History management
  // ============================================================

  const canUndo = layerHistory.past.length > 0;
  const canRedo = layerHistory.future.length > 0;

  const handleUndo = useCallback(() => {
    dispatchLayerHistory({ type: "UNDO" });
  }, []);

  const handleRedo = useCallback(() => {
    dispatchLayerHistory({ type: "REDO" });
  }, []);

  // ============================================================
  // Layer manipulation handlers
  // ============================================================

  const updateLayerState = useCallback((id: string, update: Partial<LayerState>) => {
    dispatchLayerHistory({
      type: "SET_PRESENT",
      updater: (prev) => prev.map((s) => (s.id === id ? { ...s, ...update } : s)),
    });
  }, []);

  const commitLayerState = useCallback((id: string, update: Partial<LayerState>) => {
    dispatchLayerHistory({
      type: "APPLY",
      updater: (prev) => prev.map((s) => (s.id === id ? { ...s, ...update } : s)),
    });
  }, []);

  const handleToggleVisibility = useCallback((id: string) => {
    dispatchLayerHistory({
      type: "APPLY",
      updater: (prev) =>
        prev.map((s) => (s.id === id ? { ...s, visible: !s.visible } : s)),
    });
  }, []);

  const handleToggleLock = useCallback((id: string) => {
    dispatchLayerHistory({
      type: "APPLY",
      updater: (prev) =>
        prev.map((s) => (s.id === id ? { ...s, locked: !s.locked } : s)),
    });
  }, []);

  const handleChangeOpacity = useCallback((id: string, opacity: number) => {
    dispatchLayerHistory({
      type: "APPLY",
      updater: (prev) => prev.map((s) => (s.id === id ? { ...s, opacity } : s)),
    });
  }, []);

  const handleReorderLayers = useCallback((fromIndex: number, toIndex: number) => {
    dispatchLayerHistory({
      type: "APPLY",
      updater: (prev) => {
        const next = [...prev];
        const [removed] = next.splice(fromIndex, 1);
        next.splice(toIndex, 0, removed);
        return next;
      },
    });
  }, []);

  const handleBeginTransform = useCallback((_id: string) => {
    dispatchLayerHistory({ type: "CHECKPOINT" });
  }, []);

  const handleUpdateTransform = useCallback(
    (id: string, transform: Transform2D) => {
      updateLayerState(id, { transform });
    },
    [updateLayerState]
  );

  const handleCommitTransform = useCallback(
    (id: string, transform: Transform2D) => {
      commitLayerState(id, { transform });
    },
    [commitLayerState]
  );

  const handleChangeColor = useCallback(
    (id: string, color: string | null) => {
      commitLayerState(id, { color });
    },
    [commitLayerState]
  );

  const handleResetToAuto = useCallback(
    (id: string) => {
      const original = originalLayerStates.find((s) => s.id === id);
      if (!original) return;
      commitLayerState(id, { transform: original.transform });
    },
    [originalLayerStates, commitLayerState]
  );

  const handleResetToIdentity = useCallback(
    (id: string) => {
      commitLayerState(id, { transform: createIdentityTransform() });
    },
    [commitLayerState]
  );

  const handleResetAll = useCallback(() => {
    dispatchLayerHistory({
      type: "APPLY",
      updater: () => cloneLayerStates(originalLayerStates),
    });
  }, [originalLayerStates]);

  // ============================================================
  // View handlers
  // ============================================================

  const handleZoomIn = useCallback(() => {
    setZoom((prev) => Math.min(prev * 1.2, 10));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoom((prev) => Math.max(prev / 1.2, 0.1));
  }, []);

  const handleZoomReset = useCallback(() => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, []);

  // ============================================================
  // Export handlers
  // ============================================================

  const handleExportPNG = useCallback(async () => {
    try {
      setIsExporting(true);
      const blob = await exportCanvasToPNG(layers, layerStates, canvas);
      
      // Download blob
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `overlay-${sessionId?.slice(0, 8)}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
      alert("Failed to export PNG: " + (err instanceof Error ? err.message : "Unknown error"));
    } finally {
      setIsExporting(false);
    }
  }, [layers, layerStates, canvas, sessionId]);

  const handleExportState = useCallback(() => {
    const state: CompositionState = {
      version: "1.0.0",
      created_at: new Date().toISOString(),
      canvas,
      layers: layerStates,
      alignment,
      defaults: {
        palette,
        snap_angles: snapAngles,
        snap_enabled: true,
        grid_size: 10,
      },
    };

    const json = JSON.stringify(state, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `state-${sessionId?.slice(0, 8)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [canvas, layerStates, alignment, palette, snapAngles, sessionId]);

  // ============================================================
  // Generate Differences handler
  // ============================================================

  const handleGenerateDifferences = useCallback(async () => {
    if (!sessionId) return;

    try {
      setIsGeneratingDiffs(true);
      setDifferencesError(null);
      setDifferencesResult(null);

      // Find the A layer (pair1) to get its current transform
      const aLayerState = layerStates.find((s) => {
        const layer = layers.find((l) => l.id === s.id);
        return layer?.source === "pair1";
      });

      // Find the B layer (pair2) to get its current transform
      const bLayerState = layerStates.find((s) => {
        const layer = layers.find((l) => l.id === s.id);
        return layer?.source === "pair2";
      });

      // Build transforms from current layer states
      // We now send BOTH transforms so the backend can compute the proper
      // relative transform, handling cases where either or both layers are adjusted
      let transformA: DifferencesTransform | undefined;
      let transformB: DifferencesTransform | undefined;
      
      if (aLayerState) {
        transformA = {
          scale: aLayerState.transform.scale_x,
          rotation_deg: aLayerState.transform.rotation_deg,
          translate_x: aLayerState.transform.translate_x,
          translate_y: aLayerState.transform.translate_y,
        };
      }
      
      if (bLayerState) {
        transformB = {
          scale: bLayerState.transform.scale_x,
          rotation_deg: bLayerState.transform.rotation_deg,
          translate_x: bLayerState.transform.translate_x,
          translate_y: bLayerState.transform.translate_y,
        };
      }

      const result = await generateDifferences(sessionId, transformA, transformB);
      setDifferencesResult(result);
      
      // Initialize region visibility state (all visible by default)
      if (result.regions && result.regions.length > 0) {
        setRegionVisibility(createRegionVisibilityState(result.regions));
        setAllRegionsVisible(true);
      }
    } catch (err) {
      console.error("Failed to generate differences:", err);
      setDifferencesError(err instanceof Error ? err.message : "Failed to generate differences");
    } finally {
      setIsGeneratingDiffs(false);
    }
  }, [sessionId, layerStates, layers]);

  // Region visibility handlers
  const handleToggleRegionVisibility = useCallback((regionId: number) => {
    setRegionVisibility((prev) => {
      const newState = { ...prev, [regionId]: !prev[regionId] };
      // Update allRegionsVisible based on new state
      const allVisible = Object.values(newState).every((v) => v);
      setAllRegionsVisible(allVisible);
      return newState;
    });
  }, []);

  const handleToggleAllRegions = useCallback(() => {
    const newValue = !allRegionsVisible;
    setAllRegionsVisible(newValue);
    
    if (differencesResult?.regions) {
      const newState: RegionVisibilityState = {};
      for (const region of differencesResult.regions) {
        newState[region.id] = newValue;
      }
      setRegionVisibility(newState);
    }
  }, [allRegionsVisible, differencesResult]);

  // ============================================================
  // Keyboard shortcuts
  // ============================================================

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (selectedLayerId) {
          e.preventDefault();
          setSelectedLayerId(null);
        }
        return;
      }

      // Undo/Redo
      const key = e.key.toLowerCase();
      const isModKey = e.metaKey || e.ctrlKey;
      const isUndo = isModKey && key === "z" && !e.shiftKey;
      const isRedo =
        (isModKey && key === "z" && e.shiftKey) ||
        (e.ctrlKey && !e.metaKey && key === "y");

      if (isUndo) {
        if (canUndo) {
          e.preventDefault();
          handleUndo();
        }
        return;
      }

      if (isRedo) {
        if (canRedo) {
          e.preventDefault();
          handleRedo();
        }
        return;
      }

      // Nudge selected layer
      if (selectedLayerId && !e.metaKey && !e.ctrlKey) {
        const state = layerStates.find((s) => s.id === selectedLayerId);
        if (state && !state.locked) {
          const amount = e.shiftKey ? 10 : 1;
          let dx = 0, dy = 0;

          switch (e.key) {
            case "ArrowUp": dy = -amount; break;
            case "ArrowDown": dy = amount; break;
            case "ArrowLeft": dx = -amount; break;
            case "ArrowRight": dx = amount; break;
            default: return;
          }

          e.preventDefault();
          handleCommitTransform(selectedLayerId, {
            ...state.transform,
            translate_x: state.transform.translate_x + dx,
            translate_y: state.transform.translate_y + dy,
          });
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    selectedLayerId,
    layerStates,
    canUndo,
    canRedo,
    handleUndo,
    handleRedo,
    handleCommitTransform,
  ]);

  // ============================================================
  // Render
  // ============================================================

  if (isLoading) {
    return <LoadingScreen message="Loading composition..." />;
  }

  if (error) {
    return <ErrorScreen message={error} />;
  }

  // Get selected layer info
  const selectedLayer = selectedLayerId
    ? layers.find((l) => l.id === selectedLayerId) || null
    : null;
  const selectedLayerState = selectedLayerId
    ? layerStates.find((s) => s.id === selectedLayerId) || null
    : null;

  const isDirty = JSON.stringify(layerStates) !== JSON.stringify(originalLayerStates);

  return (
    <div className="h-screen flex flex-col bg-ink-950">
      {/* Header */}
      <header className="h-14 bg-ink-900 border-b border-ink-700 flex items-center px-4">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blueprint-500 to-blueprint-700 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h1 className="text-lg font-bold text-ink-100">Overlay Editor</h1>
        </Link>
        <span className="ml-4 text-sm text-ink-500">
          Session: {sessionId?.slice(0, 8)}...
        </span>
      </header>

      {/* Toolbar */}
      <Toolbar
        sessionId={sessionId || ""}
        status={status}
        confidence={confidence}
        warnings={warnings}
        zoom={zoom}
        canUndo={canUndo}
        canRedo={canRedo}
        isDirty={isDirty}
        isExporting={isExporting}
        onUndo={handleUndo}
        onRedo={handleRedo}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onZoomReset={handleZoomReset}
        onExportPNG={handleExportPNG}
        onExportState={handleExportState}
        onResetAll={handleResetAll}
      />

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel - Layers */}
        <LayerPanel
          layers={layers}
          layerStates={layerStates}
          selectedLayerId={selectedLayerId}
          fileNameA={fileNameA}
          fileNameB={fileNameB}
          onSelectLayer={setSelectedLayerId}
          onToggleVisibility={handleToggleVisibility}
          onToggleLock={handleToggleLock}
          onChangeOpacity={handleChangeOpacity}
          onReorderLayers={handleReorderLayers}
        />

        {/* Center - Canvas */}
        <CanvasRenderer
          layers={layers}
          layerStates={layerStates}
          canvas={canvas}
          selectedLayerId={selectedLayerId}
          zoom={zoom}
          panX={panX}
          panY={panY}
          onSelectLayer={setSelectedLayerId}
          onBeginTransform={handleBeginTransform}
          onUpdateTransform={handleUpdateTransform}
          onZoomChange={setZoom}
          onPanChange={(x, y) => { setPanX(x); setPanY(y); }}
          regions={differencesResult?.regions || []}
          regionVisibility={regionVisibility}
        />

        {/* Right panel - Properties and Differences */}
        <div className="w-72 bg-ink-900 border-l border-ink-700 flex flex-col h-full overflow-hidden">
          {/* Properties Panel Content */}
          <div className="flex-1 overflow-y-auto">
            <PropertiesPanel
              selectedLayer={selectedLayer}
              selectedLayerState={selectedLayerState}
              alignment={alignment}
              palette={palette}
              snapAngles={snapAngles}
              onUpdateTransform={handleCommitTransform}
              onChangeColor={handleChangeColor}
              onResetToAuto={handleResetToAuto}
              onResetToIdentity={handleResetToIdentity}
            />
          </div>

          {/* Smart Analysis Panel */}
          <div className="border-t border-ink-700 p-4 flex-shrink-0">
            <h3 className="text-sm font-semibold text-ink-200 uppercase tracking-wider mb-3">
              Smart Analysis
            </h3>
            
            <button
              onClick={handleGenerateDifferences}
              disabled={isGeneratingDiffs}
              className={`
                w-full px-4 py-2 rounded font-medium text-sm
                ${isGeneratingDiffs 
                  ? "bg-ink-700 text-ink-400 cursor-not-allowed" 
                  : "bg-blueprint-600 hover:bg-blueprint-500 text-white"}
              `}
            >
              {isGeneratingDiffs ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-ink-400 border-t-transparent rounded-full animate-spin" />
                  Analyzing...
                </span>
              ) : (
                "Analyze Differences"
              )}
            </button>

            {differencesError && (
              <div className="mt-3 p-3 bg-red-900/30 border border-red-700 rounded text-sm text-red-300">
                {differencesError}
              </div>
            )}

            {differencesResult && (
              <div className="mt-3 space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2 text-ink-400">
                    <span className={`w-2 h-2 rounded-full ${differencesResult.ai_available ? "bg-green-500" : "bg-yellow-500"}`} />
                    {differencesResult.ai_available 
                      ? (differencesResult.model_display_name || "AI Generated")
                      : "Not Available"}
                  </div>
                  {differencesResult.ai_available && differencesResult.is_vlm && (
                    <span className="text-ink-500" title="Vision Language Model">VLM</span>
                  )}
                </div>
                
                <div className="p-3 bg-ink-800 rounded text-sm text-ink-200 leading-relaxed max-h-48 overflow-y-auto">
                  {differencesResult.summary}
                </div>
                
                <div className="flex justify-between text-xs text-ink-500">
                  <span>{differencesResult.total_regions} region{differencesResult.total_regions !== 1 ? "s" : ""}</span>
                  <span>{differencesResult.diff_percentage.toFixed(1)}% diff</span>
                </div>
              </div>
            )}
          </div>

          {/* Region Annotations Panel - shown after analysis */}
          {differencesResult && differencesResult.regions.length > 0 && (() => {
            // Get layer colors for contrasting annotation color
            const layerColors = layerStates
              .filter(s => s.visible)
              .map(s => s.color)
              .filter(Boolean) as string[];
            
            return (
              <RegionAnnotationsPanel
                regions={differencesResult.regions}
                regionVisibility={regionVisibility}
                allRegionsVisible={allRegionsVisible}
                onToggleRegion={handleToggleRegionVisibility}
                onToggleAll={handleToggleAllRegions}
                colorA={layerColors[0] || "red"}
                colorB={layerColors[1] || "cyan"}
              />
            );
          })()}
        </div>
      </div>
    </div>
  );
}

// ============================================================
// Loading/Error Screens
// ============================================================

function LoadingScreen({ message }: { message: string }) {
  return (
    <div className="h-screen flex items-center justify-center bg-ink-950">
      <div className="text-center">
        <div className="w-12 h-12 border-4 border-blueprint-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-ink-400">{message}</p>
      </div>
    </div>
  );
}

function ErrorScreen({ message }: { message: string }) {
  return (
    <div className="h-screen flex items-center justify-center bg-ink-950">
      <div className="text-center max-w-md p-8 bg-ink-900 rounded-lg border border-ink-700">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-500/20 flex items-center justify-center">
          <svg className="w-8 h-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-ink-100 mb-2">Error</h2>
        <p className="text-ink-400 mb-6">{message}</p>
        <Link
          href="/"
          className="inline-block px-6 py-2 bg-blueprint-600 hover:bg-blueprint-500 rounded text-white"
        >
          Go to Home
        </Link>
      </div>
    </div>
  );
}
