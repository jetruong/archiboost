"use client";

/**
 * Layer Panel Component
 * 
 * Left sidebar for managing layers:
 * - Layer list with drag/drop reordering
 * - Visibility toggle
 * - Lock toggle
 * - Opacity slider
 */

import React from "react";
import {
  LayerInfo,
  LayerState,
} from "@/lib/overlay-types";

// Map color names to hex values for display
const COLOR_NAME_TO_HEX: Record<string, string> = {
  red: "#ff0000",
  cyan: "#00ffff",
  blue: "#0000ff",
  green: "#00ff00",
  magenta: "#ff00ff",
  yellow: "#ffff00",
  black: "#000000",
  white: "#ffffff",
};

/**
 * Get the hex color for display (handles both names and hex values)
 */
function getDisplayColor(color: string): string {
  if (color.startsWith("#")) return color;
  return COLOR_NAME_TO_HEX[color.toLowerCase()] || color;
}

interface LayerPanelProps {
  layers: LayerInfo[];
  layerStates: LayerState[];
  selectedLayerId: string | null;
  fileNameA?: string;
  fileNameB?: string;
  onSelectLayer: (id: string | null) => void;
  onToggleVisibility: (id: string) => void;
  onToggleLock: (id: string) => void;
  onChangeOpacity: (id: string, opacity: number) => void;
  onReorderLayers: (fromIndex: number, toIndex: number) => void;
}

export function LayerPanel({
  layers,
  layerStates,
  selectedLayerId,
  fileNameA = "Drawing A",
  fileNameB = "Drawing B",
  onSelectLayer,
  onToggleVisibility,
  onToggleLock,
  onChangeOpacity,
  onReorderLayers,
}: LayerPanelProps) {
  // Get layer state by ID
  const getState = (id: string): LayerState | undefined =>
    layerStates.find((s) => s.id === id);

  // Get layer info by ID
  const getInfo = (id: string): LayerInfo | undefined =>
    layers.find((l) => l.id === id);

  // Build ordered list (layer states determine order)
  const orderedLayers = layerStates
    .map((state) => ({
      state,
      info: getInfo(state.id),
    }))
    .filter((item) => item.info !== undefined);

  return (
    <div className="w-64 bg-ink-900 border-r border-ink-700 flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-ink-700">
        <h2 className="text-sm font-semibold text-ink-200 uppercase tracking-wider">
          Layers
        </h2>
      </div>

      {/* Layer List */}
      <div className="flex-1 overflow-y-auto">
        {orderedLayers.map(({ state, info }, index) => {
          if (!info) return null;
          const isSelected = state.id === selectedLayerId;
          const sourceColor =
            info.source === "pair1" ? "bg-red-500" : "bg-cyan-500";

          return (
            <div
              key={state.id}
              className={`
                border-b border-ink-800 p-3 cursor-pointer transition-colors
                ${isSelected ? "bg-ink-800" : "hover:bg-ink-850"}
              `}
              onClick={() => onSelectLayer(state.id)}
            >
              {/* Layer Header */}
              <div className="flex items-center gap-2 mb-2">
                {/* Source indicator */}
                <div
                  className={`w-2 h-2 rounded-full ${sourceColor}`}
                  title={info.source === "pair1" ? fileNameA : fileNameB}
                />

                {/* Name */}
                <span
                  className={`flex-1 text-sm truncate ${
                    state.visible ? "text-ink-200" : "text-ink-500"
                  }`}
                  title={info.source === "pair1" ? fileNameA : fileNameB}
                >
                  {info.source === "pair1" ? fileNameA : fileNameB}
                </span>

                {/* Visibility toggle */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleVisibility(state.id);
                  }}
                  className={`p-1 rounded hover:bg-ink-700 ${
                    state.visible ? "text-ink-300" : "text-ink-600"
                  }`}
                  title={state.visible ? "Hide layer" : "Show layer"}
                >
                  {state.visible ? (
                    <EyeIcon className="w-4 h-4" />
                  ) : (
                    <EyeSlashIcon className="w-4 h-4" />
                  )}
                </button>

                {/* Lock toggle */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleLock(state.id);
                  }}
                  className={`p-1 rounded hover:bg-ink-700 ${
                    state.locked ? "text-yellow-500" : "text-ink-600"
                  }`}
                  title={state.locked ? "Unlock layer" : "Lock layer"}
                >
                  {state.locked ? (
                    <LockClosedIcon className="w-4 h-4" />
                  ) : (
                    <LockOpenIcon className="w-4 h-4" />
                  )}
                </button>
              </div>

              {/* Opacity slider */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-ink-500 w-12">Opacity</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={state.opacity}
                  onChange={(e) =>
                    onChangeOpacity(state.id, parseFloat(e.target.value))
                  }
                  onClick={(e) => e.stopPropagation()}
                  className="flex-1 h-1 bg-ink-700 rounded-lg appearance-none cursor-pointer"
                />
                <span className="text-xs text-ink-400 w-8 text-right">
                  {Math.round(state.opacity * 100)}%
                </span>
              </div>

              {/* Color indicator */}
              {state.color && (
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-xs text-ink-500 w-12">Color</span>
                  <div
                    className="w-4 h-4 rounded border border-ink-600"
                    style={{ backgroundColor: getDisplayColor(state.color) }}
                  />
                  <span className="text-xs text-ink-400">{state.color}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer with layer count */}
      <div className="px-4 py-2 border-t border-ink-700 text-xs text-ink-500">
        {layers.length} layers
      </div>
    </div>
  );
}

// ============================================================
// Icon Components
// ============================================================

function EyeIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
      />
    </svg>
  );
}

function EyeSlashIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"
      />
    </svg>
  );
}

function LockClosedIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
      />
    </svg>
  );
}

function LockOpenIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z"
      />
    </svg>
  );
}
