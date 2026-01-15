"use client";

/**
 * Properties Panel Component
 * 
 * Right sidebar for selected layer properties:
 * - Transform controls (translate, scale, rotate)
 * - Color picker
 * - Reset to defaults
 * - Snap controls
 */

import React, { useState } from "react";
import {
  LayerInfo,
  LayerState,
  Transform2D,
  createIdentityTransform,
  AlignmentState,
  ColorPalette,
} from "@/lib/overlay-types";

// Map color names to hex values for the color picker input
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
 * Convert a color (name or hex) to hex format for the color picker
 */
function toHex(color: string | null | undefined): string {
  if (!color) return "#ffffff";
  // If it's already hex format, return as-is
  if (color.startsWith("#")) return color;
  // Try to look up the color name
  return COLOR_NAME_TO_HEX[color.toLowerCase()] || "#ffffff";
}

interface PropertiesPanelProps {
  selectedLayer: LayerInfo | null;
  selectedLayerState: LayerState | null;
  alignment: AlignmentState;
  palette: ColorPalette;
  snapAngles: number[];
  onUpdateTransform: (id: string, transform: Transform2D) => void;
  onChangeColor: (id: string, color: string | null) => void;
  onResetToAuto: (id: string) => void;
  onResetToIdentity: (id: string) => void;
}

export function PropertiesPanel({
  selectedLayer,
  selectedLayerState,
  alignment,
  palette,
  snapAngles,
  onUpdateTransform,
  onChangeColor,
  onResetToAuto,
  onResetToIdentity,
}: PropertiesPanelProps) {
  const [lockAspectRatio, setLockAspectRatio] = useState(true);

  if (!selectedLayer || !selectedLayerState) {
    return (
      <div className="flex flex-col h-full">
        <div className="px-4 py-3 border-b border-ink-700">
          <h2 className="text-sm font-semibold text-ink-200 uppercase tracking-wider">
            Properties
          </h2>
        </div>
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-ink-500 text-sm text-center">
            Select a layer to view properties
          </p>
        </div>
      </div>
    );
  }

  const transform = selectedLayerState.transform;

  // Handle transform value change
  const handleTransformChange = (
    field: keyof Transform2D,
    value: number
  ) => {
    const newTransform = { ...transform, [field]: value };

    // If aspect ratio locked and scaling, adjust the other scale
    if (lockAspectRatio && field === "scale_x") {
      newTransform.scale_y = value;
    } else if (lockAspectRatio && field === "scale_y") {
      newTransform.scale_x = value;
    }

    onUpdateTransform(selectedLayerState.id, newTransform);
  };

  // Snap rotation to nearest snap angle
  const handleSnapRotation = () => {
    const current = transform.rotation_deg;
    const normalized = ((current % 360) + 360) % 360;
    
    let closest = 0;
    let minDiff = Infinity;
    
    for (const angle of snapAngles) {
      const diff = Math.abs(normalized - angle);
      const wrapDiff = Math.min(diff, 360 - diff);
      if (wrapDiff < minDiff) {
        minDiff = wrapDiff;
        closest = angle;
      }
    }
    
    handleTransformChange("rotation_deg", closest);
  };

  // Quick rotation buttons
  const handleQuickRotate = (degrees: number) => {
    handleTransformChange("rotation_deg", degrees);
  };

  // Flip handlers
  const handleFlipH = () => {
    handleTransformChange("scale_x", -transform.scale_x);
  };

  const handleFlipV = () => {
    handleTransformChange("scale_y", -transform.scale_y);
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      {/* Header */}
      <div className="px-4 py-3 border-b border-ink-700">
        <h2 className="text-sm font-semibold text-ink-200 uppercase tracking-wider">
          Properties
        </h2>
        <p className="text-xs text-ink-500 mt-1 truncate">{selectedLayer.name}</p>
      </div>

      {/* Transform Section */}
      <div className="p-4 border-b border-ink-700">
        <h3 className="text-xs font-semibold text-ink-300 uppercase tracking-wider mb-3">
          Transform
        </h3>

        {/* Position */}
        <div className="grid grid-cols-2 gap-2 mb-3">
          <div>
            <label className="text-xs text-ink-500 block mb-1">X</label>
            <input
              type="number"
              value={Math.round(transform.translate_x)}
              onChange={(e) =>
                handleTransformChange("translate_x", parseFloat(e.target.value) || 0)
              }
              className="w-full bg-ink-800 border border-ink-700 rounded px-2 py-1 text-sm text-ink-200"
              step="1"
            />
          </div>
          <div>
            <label className="text-xs text-ink-500 block mb-1">Y</label>
            <input
              type="number"
              value={Math.round(transform.translate_y)}
              onChange={(e) =>
                handleTransformChange("translate_y", parseFloat(e.target.value) || 0)
              }
              className="w-full bg-ink-800 border border-ink-700 rounded px-2 py-1 text-sm text-ink-200"
              step="1"
            />
          </div>
        </div>

        {/* Scale */}
        <div className="grid grid-cols-2 gap-2 mb-3">
          <div>
            <label className="text-xs text-ink-500 block mb-1">Scale X</label>
            <input
              type="number"
              value={transform.scale_x.toFixed(3)}
              onChange={(e) =>
                handleTransformChange("scale_x", parseFloat(e.target.value) || 1)
              }
              className="w-full bg-ink-800 border border-ink-700 rounded px-2 py-1 text-sm text-ink-200"
              step="0.01"
            />
          </div>
          <div>
            <label className="text-xs text-ink-500 block mb-1">Scale Y</label>
            <input
              type="number"
              value={transform.scale_y.toFixed(3)}
              onChange={(e) =>
                handleTransformChange("scale_y", parseFloat(e.target.value) || 1)
              }
              className="w-full bg-ink-800 border border-ink-700 rounded px-2 py-1 text-sm text-ink-200"
              step="0.01"
            />
          </div>
        </div>

        {/* Lock aspect ratio */}
        <div className="flex items-center gap-2 mb-3">
          <input
            type="checkbox"
            id="lockAspect"
            checked={lockAspectRatio}
            onChange={(e) => setLockAspectRatio(e.target.checked)}
            className="w-4 h-4 bg-ink-800 border-ink-700 rounded"
          />
          <label htmlFor="lockAspect" className="text-xs text-ink-400">
            Lock aspect ratio
          </label>
        </div>

        {/* Rotation */}
        <div className="mb-3">
          <label className="text-xs text-ink-500 block mb-1">Rotation (°)</label>
          <div className="flex gap-2">
            <input
              type="number"
              value={Math.round(transform.rotation_deg)}
              onChange={(e) =>
                handleTransformChange("rotation_deg", parseFloat(e.target.value) || 0)
              }
              className="flex-1 bg-ink-800 border border-ink-700 rounded px-2 py-1 text-sm text-ink-200"
              step="1"
            />
            <button
              onClick={handleSnapRotation}
              className="px-2 py-1 bg-ink-700 hover:bg-ink-600 rounded text-xs text-ink-300"
              title="Snap to nearest 90°"
            >
              Snap
            </button>
          </div>
        </div>

        {/* Quick rotation buttons */}
        <div className="flex gap-1 mb-3">
          {snapAngles.map((angle) => (
            <button
              key={angle}
              onClick={() => handleQuickRotate(angle)}
              className={`flex-1 px-2 py-1 text-xs rounded ${
                Math.abs(transform.rotation_deg - angle) < 1
                  ? "bg-blueprint-600 text-white"
                  : "bg-ink-800 hover:bg-ink-700 text-ink-300"
              }`}
            >
              {angle}°
            </button>
          ))}
        </div>

        {/* Flip buttons */}
        <div className="flex gap-2">
          <button
            onClick={handleFlipH}
            className="flex-1 px-3 py-2 bg-ink-800 hover:bg-ink-700 rounded text-xs text-ink-300"
          >
            ↔ Flip H
          </button>
          <button
            onClick={handleFlipV}
            className="flex-1 px-3 py-2 bg-ink-800 hover:bg-ink-700 rounded text-xs text-ink-300"
          >
            ↕ Flip V
          </button>
        </div>
      </div>

      {/* Color Section */}
      <div className="p-4 border-b border-ink-700">
        <h3 className="text-xs font-semibold text-ink-300 uppercase tracking-wider mb-3">
          Color
        </h3>

        {/* Palette colors */}
        <div className="grid grid-cols-4 gap-2 mb-3">
          {Object.entries(palette.colors).map(([name, hex]) => (
            <button
              key={name}
              onClick={() => onChangeColor(selectedLayerState.id, name)}
              className={`w-8 h-8 rounded border-2 ${
                selectedLayerState.color?.toLowerCase() === name.toLowerCase() ||
                selectedLayerState.color?.toLowerCase() === hex.toLowerCase()
                  ? "border-white"
                  : "border-transparent hover:border-ink-600"
              }`}
              style={{ backgroundColor: hex }}
              title={name}
            />
          ))}
        </div>

        {/* Custom color */}
        <div className="flex items-center gap-2">
          <label className="text-xs text-ink-500">Custom:</label>
          <input
            type="color"
            value={toHex(selectedLayerState.color)}
            onChange={(e) => onChangeColor(selectedLayerState.id, e.target.value)}
            className="w-8 h-8 bg-transparent border-0 cursor-pointer"
          />
          <button
            onClick={() => onChangeColor(selectedLayerState.id, null)}
            className="ml-auto text-xs text-ink-500 hover:text-ink-300"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Reset Section */}
      <div className="p-4">
        <h3 className="text-xs font-semibold text-ink-300 uppercase tracking-wider mb-3">
          Reset
        </h3>

        <div className="space-y-2">
          {/* Reset to auto-aligned position */}
          {selectedLayer.source === "pair2" && alignment.confidence > 0 && (
            <button
              onClick={() => onResetToAuto(selectedLayerState.id)}
              className="w-full px-3 py-2 bg-blueprint-600 hover:bg-blueprint-500 rounded text-sm text-white"
            >
              Reset to Auto-Aligned
            </button>
          )}

          {/* Reset to identity */}
          <button
            onClick={() => onResetToIdentity(selectedLayerState.id)}
            className="w-full px-3 py-2 bg-ink-800 hover:bg-ink-700 rounded text-sm text-ink-300"
          >
            Reset to Identity
          </button>
        </div>

        {/* Alignment info */}
        {alignment.confidence > 0 && (
          <div className="mt-4 p-3 bg-ink-800 rounded">
            <p className="text-xs text-ink-400 mb-1">Auto-alignment confidence:</p>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-ink-700 rounded overflow-hidden">
                <div
                  className={`h-full ${
                    alignment.confidence > 0.7
                      ? "bg-green-500"
                      : alignment.confidence > 0.4
                      ? "bg-yellow-500"
                      : "bg-red-500"
                  }`}
                  style={{ width: `${alignment.confidence * 100}%` }}
                />
              </div>
              <span className="text-xs text-ink-300">
                {Math.round(alignment.confidence * 100)}%
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
